# Written by Yukang Chen
# Some code based on https://github.com/epfml/landmark-attention
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
import os
import copy
import json
import math
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import transformers
from torch.utils.data import Dataset
from transformers import Trainer, DataCollatorForLanguageModeling, PreTrainedModel, LlamaConfig, LlamaModel, LlamaForCausalLM
from peft import LoraConfig, get_peft_model
from torch.distributed import barrier
import ipdb
from colbert.infra import ColBERTConfig
from colbert.training.rerank_batcher import RerankBatcher

from colbert.utils.amp import MixedPrecisionManager
from colbert.training.lazy_batcher import LazyBatcher
from colbert.parameters import DEVICE

from colbert.modeling.colbert import ColBERT
from colbert.modeling.reranker.electra import ElectraReranker

from colbert.utils.utils import print_message
from colbert.training.utils import print_progress, manage_checkpoints
from colbert.modeling.reranker.tokenizer import RerankerTokenizer
from typing import List, Optional, Tuple, Union

from colbert.modeling.tokenization import QueryTokenizer, DocTokenizer, tensorize_triples
from colbert.modeling.colbert import colbert_score
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer
from colbert.data import Queries
from colbert import Searcher
import csv
from concurrent.futures import ProcessPoolExecutor

from llama_ir2 import LlamaIRForCausalLM
from IRTrainer import LlamaIRTrainer
from transformers.optimization import get_scheduler
import datetime
from peft import LoraConfig, get_peft_model, PeftModel
torch.distributed.init_process_group('nccl', init_method=None, timeout=datetime.timedelta(seconds=3600), world_size=- 1, rank=- 1, store=None, group_name='', pg_options=None)


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
DEFAULT_DOC_START_TOKEN = "<doc_start>"
DEFAULT_DOC_END_TOKEN = "<doc_end>"

def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f

def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}



@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="EleutherAI/pythia-1.4b-deduped")
    colbert_path: Optional[str] = field(default=None)
    peft_path: Optional[str] = field(default=None)
    model_type: Optional[str] = field(default="llama")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=4048,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    use_flash_attn: bool = field(
        default=True,
        metadata={"help": "Whether use flash attention for training."},
    )
    low_rank_training: bool = field(
        default=True,
        metadata={"help": "Whether use low rank adaptation for training."},
    )
    trainable_params: str = field(
        default="embed,norm",
        metadata={"help": "Additional trainable parameters except LoRA weights, if low rank training."},
    )
    lr_colbert: float = field(
        default=3e-6,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def process_texts(strings, tokenizer):
    results = []
    
    with ProcessPoolExecutor(max_workers=32) as executor:
        futures = [executor.submit(_tokenize_fn, strings, tokenizer) for text in texts]
        
        for future in futures:
            results.append(future.result())
    
    return results

def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    doc: Sequence[Sequence[str]],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + ' ' + t for s, t in zip(sources, targets)]
    doc = [[f'\nReference:\n' + word for word in sublist] for sublist in doc]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    
    input_ids = examples_tokenized["input_ids"]
    sources_tokenized = sources_tokenized["input_ids"]
    doc_input_ids = []
    
    for i in range(len(doc)):
        doc_input_ids.append(_tokenize_fn(doc[i], tokenizer)["input_ids"])
    

    #tokenizer.decode(labels[0][:len(sources_tokenized[0])])

    labels = copy.deepcopy(input_ids)
    for i, label in enumerate(labels):
        label[:len(sources_tokenized[i])] = IGNORE_INDEX
    
    return dict(input_ids=input_ids, labels=labels, doc_input_ids=doc_input_ids)


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, doc_tokenizer, query_tokenizer):
        super(LazySupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = jload(data_path)
        self.list_data_dict = list_data_dict
        """
        list_data_dict = []
        for x in list_data_dict2:
            if len(x['output']) > 2:
                continue
            list_data_dict.append(x)
        """
        logging.warning("Formatting inputs...")
        '''
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        '''

        #sources = [example["instruction"] for example in list_data_dict]
        self.sources = ['\nMy Question:\n' + example["input"] for example in self.list_data_dict]
        

        self.targets = [f"{example['output']}{tokenizer.eos_token}" for example in self.list_data_dict]
        
        self.tokenizer = tokenizer

        self.query_tokenizer = query_tokenizer
        self.doc_tokenizer = doc_tokenizer

        

        self.doc = [example["doc"] for example in self.list_data_dict]
        
    def __len__(self):
        return len(self.sources)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        data_dict = preprocess([self.sources[i]], [self.targets[i]], [self.doc[i]], self.tokenizer)
        
        input_ids = data_dict["input_ids"]
        labels = data_dict["labels"]
        doc_token = self.doc_tokenizer.tensorize(self.doc[i])
        query_token = self.query_tokenizer.tensorize([self.list_data_dict[i]["input"]])
        doc_input_ids = data_dict["doc_input_ids"]
        return dict(input_ids=input_ids[0], labels=labels[0], doc_token=doc_token, query_token=query_token, 
                    doc_input_ids=doc_input_ids[0])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, doc_token, query_token, doc_input_ids = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels", "doc_token", "query_token", "doc_input_ids"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            doc_input_ids=doc_input_ids,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            doc_token=doc_token,
            query_token=query_token
        )

def split_strings(input_list, max_length=400):
    """
    Split strings in the list to ensure each string is no longer than the specified max_length.

    :param input_list: List of strings.
    :param max_length: Maximum length of each split string in words.
    :return: List of strings where each string is no longer than max_length.
    """
    split_list = []
    
    for text in input_list:
        words = text.split()
        for i in range(0, len(words), max_length):
            split_list.append(" ".join(words[i:i + max_length]))
    
    return split_list


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, doc_tokenizer, query_tokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer, doc_tokenizer=doc_tokenizer, query_tokenizer=query_tokenizer, data_path=data_args.data_path)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    return list(lora_module_names)




def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # NOTE: May expand supported model types in the future
    """
    if model_args.model_type == "gpt-neox":
        replace_gpt_neox_attn(training_args.use_flash_attn) 
    else:
        replace_llama_attn(training_args.use_flash_attn)
    """
    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    """
    orig_rope_scaling = getattr(config, "rope_scaling", None)
    if orig_rope_scaling is None:
        orig_rope_scaling = {"factor": 1}
    orig_rope_scaling_factor = orig_rope_scaling["factor"] if "factor" in orig_rope_scaling.keys() else 1
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len:
        orig_ctx_len *= orig_rope_scaling_factor
        if training_args.model_max_length > orig_ctx_len:
            scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
            config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    """
    colbert_config = ColBERTConfig(
        doc_maxlen=128, 
        query_maxlen=512,
        bsize=4,
        lr=3e-6, 
        accumsteps=1,
        use_ib_negatives=False
    )
    colbert_config.checkpoint = model_args.colbert_path
    colbert_config.bsize = colbert_config.bsize // colbert_config.nranks

    

    doc_tokenizer = DocTokenizer(colbert_config)
    query_tokenizer = QueryTokenizer(colbert_config)

    # Load model and tokenizer
    model = LlamaIRForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch.bfloat16,
    )

    model.model.initialize_colbert(colbert_config)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN




    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    model.tokenizer = tokenizer


    
    data_module = make_supervised_data_module(tokenizer=tokenizer, doc_tokenizer=doc_tokenizer, query_tokenizer=query_tokenizer, data_args=data_args)
    

    if training_args.low_rank_training and not model_args.peft_path:
        if model_args.model_type == "gpt-neox":
            # added `dense` to match with llama as the basic LoRA would only target 'query_key_value'
            targets = ["query_key_value", "dense"]
        else:
            targets=['q_proj','k_proj','v_proj','o_proj','gate_proj','down_proj','up_proj','lm_head']

        config = LoraConfig(
            r=128,
            lora_alpha=256,
            target_modules=targets,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)
        # enable trainable params
    else:
        trainable_params = os.path.join(model_args.peft_path, "trainable_params.bin")
        if os.path.isfile(trainable_params):
            model.load_state_dict(torch.load(trainable_params, map_location=model.device), strict=False)
        model = PeftModel.from_pretrained(
                model,
                model_args.peft_path,
                device_map="auto",
                torch_dtype=torch.float16,
                is_trainable=True
        )

    [p.requires_grad_() for n, p in model.named_parameters() if any([k in n for k in training_args.trainable_params.split(",")])]
        
    

    model.config.use_cache = False         # required for gradient checkpointing
    model.enable_input_require_grads()     # required for gradient checkpointing
    model.gradient_checkpointing_enable()  # enable gradient checkpointing
        
    trainer = LlamaIRTrainer(model=model, args=training_args, **data_module)
    
    trainer.train(resume_from_checkpoint=model_args.peft_path)
    #trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
