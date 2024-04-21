


import os
import sys
import math
import torch
import argparse
import textwrap
import transformers
from peft import PeftModel
from transformers import GenerationConfig, TextStreamer
from llama_attn_replace import replace_llama_attn
import json
import io
import ipdb
from typing import Optional, Dict, Sequence
from peft import (
    prepare_model_for_int8_training,
    get_peft_model,
    get_peft_model_state_dict,
    PeftModel,
    PeftConfig,
)
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
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from colbert.modeling.colbert import colbert_score
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer
from colbert.data import Queries
from colbert import Searcher
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_DOC_START_TOKEN = "<doc_start>"
DEFAULT_DOC_END_TOKEN = "<doc_end>"
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
def colbert_score_reduce(scores_padded, D_mask, config: ColBERTConfig):
    D_padding = ~D_mask.view(scores_padded.size(0), scores_padded.size(1)).bool()
    scores_padded[D_padding] = -9999
    scores = scores_padded.max(1).values

    assert config.interaction in ['colbert', 'flipr'], config.interaction

    if config.interaction == 'flipr':
        assert config.query_maxlen == 64, ("for now", config)
        # assert scores.size(1) == config.query_maxlen, scores.size()

        K1 = config.query_maxlen // 2
        K2 = 8

        A = scores[:, :config.query_maxlen].topk(K1, dim=-1).values.sum(-1)
        B = 0

        if K2 <= scores.size(1) - config.query_maxlen:
            B = scores[:, config.query_maxlen:].topk(K2, dim=-1).values.sum(1)

        return A + B

    return scores.sum(-1)


# TODO: Wherever this is called, pass `config=`
def colbert_score(Q, D_padded, D_mask, config=ColBERTConfig()):
    """
        Supply sizes Q = (1 | num_docs, *, dim) and D = (num_docs, *, dim).
        If Q.size(0) is 1, the matrix will be compared with all passages.
        Otherwise, each query matrix will be compared against the *aligned* passage.

        EVENTUALLY: Consider masking with -inf for the maxsim (or enforcing a ReLU).
    """

    use_gpu = config.total_visible_gpus > 0
    if use_gpu:
        Q, D_padded, D_mask = Q.cuda(), D_padded.cuda(), D_mask.cuda()

    assert Q.dim() == 3, Q.size()
    assert D_padded.dim() == 3, D_padded.size()
    assert Q.size(0) in [1, D_padded.size(0)]

    scores = D_padded @ Q.to(dtype=D_padded.dtype).permute(0, 2, 1)

    return colbert_score_reduce(scores, D_mask, config)




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
Prompt = """
You should ask these questions and here are some examples:
A female newborn delivered at 37 weeks\u2019 gestation develops respiratory distress immediately after birth. She was delivered vaginally to a 31-year-old woman, gravida 1, para 1. Pregnancy was complicated by gestational diabetes mellitus treated with insulin during the third trimester. The newborn's pulse is 136/min, respirations are 57/min, and blood pressure is 60/35 mm Hg. Pulse oximetry on room air shows an oxygen saturation of 91% when the newborn is crying and a saturation of 85% at rest. Examination shows grunting breath sounds and perioral blue discoloration that improves when the patient cries. Lungs are clear to auscultation. Cardiac examination shows a split S2 during inspiration but no murmurs are heard. Femoral pulses are palpable bilaterally. A suction catheter cannot be passed through the nares. In addition to establishing an oral airway, which of the following is the most appropriate treatment for this patient's condition?\nA\nArterial switch procedure\nB\nEndoscopic resection of the posterior nasal septum\nC\nReconnection of the upper esophageal pouch with the lower esophagus\nD\nAnastomosis between subclavian and pulmonary artery\nE\nEndotracheal administration of artificial surfactant
Correct Answer: B
A 2-year-old girl is brought to the physician by her parents because of clumsiness and difficulty walking. She began to walk at 12 months and continues to have difficulty standing still without support. She also appears to have difficulty grabbing objects in front of her. Over the past year, she has had 5 episodes of sinusitis requiring antibiotic treatment and was hospitalized twice for bacterial pneumonia. Physical examination shows an unstable, narrow-based gait and several hyperpigmented skin patches. Serum studies show decreased levels of IgA and IgG and an increased level of alpha-fetoprotein. Over the next 5 years, which of the following complications is this patient most likely to develop?\nA\nChronic eczema\nB\nConjunctival telangiectasias\nC\nPes cavus\nD\nCardiac rhabdomyoma\nE\nCeliac disease\nF\nChronic lymphocytic leukemia
Correct Answer: B
"""

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--base_model', type=str, default="/data1/pretrained-models/llama-7b-hf")
    parser.add_argument('--colbert_model', type=str, default="/home/jwang/Project/doctorrobot/LongLoRA/colbert/colbertv2.0")
    parser.add_argument('--data_path', type=str, default="/home/jwang/Project/doctorrobot/LongLoRA/colbert/colbertv2.0")
    parser.add_argument('--peft_path', type=str, default=None)
    parser.add_argument('--cache_dir', type=str, default="./cache")
    parser.add_argument('--context_size', type=int, default=-1, help='context size during fine-tuning')
    parser.add_argument('--temperature', type=float, default=0.6, help='')
    parser.add_argument('--top_p', type=float, default=0.9, help='')
    parser.add_argument('--max_gen_len', type=int, default=512, help='')
    args = parser.parse_args()
    return args

def read_txt_file(material_txt):
    if not material_txt.split(".")[-1]=='txt':
        raise ValueError("Only support txt or pdf file.")
    content = ""
    with open(material_txt) as f:
        for line in f.readlines():
            content += line
    return content

def build_generator(
    model, tokenizer, temperature=0.6, top_p=0.9, max_gen_len=4096, use_cache=True
):
    def response(prompt):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        streamer = TextStreamer(tokenizer)
        
        output = model.generate(
            **inputs,
            max_new_tokens=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            use_cache=use_cache,
            streamer=streamer,
        )
        
        out = tokenizer.decode(output[0], skip_special_tokens=True)

        out = out.split(prompt.lstrip("<s>"))[1].strip()
        return out

    return response
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

def main(args):

    
    config = transformers.AutoConfig.from_pretrained(
        args.base_model,
        cache_dir=args.cache_dir,
    )
    # Set RoPE scaling factor
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and args.context_size > orig_ctx_len:
        scaling_factor = float(math.ceil(args.context_size / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    # Load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.base_model,
        config=config,
        cache_dir=args.cache_dir,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    model.resize_token_embeddings(32001)
        
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.base_model,
        cache_dir=args.cache_dir,
        model_max_length=args.context_size if args.context_size > orig_ctx_len else orig_ctx_len,
        padding_side="right",
        use_fast=False,
    )
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
        tokenizer=tokenizer,
        model=model,
    )
    model.eval()

    

    respond = build_generator(model, tokenizer, temperature=args.temperature, top_p=args.top_p,
                              max_gen_len=args.max_gen_len, use_cache=True)
    

    sources = []

    list_data_dict = jload(args.data_path)

    doc = [example["doc"] for example in list_data_dict]

    colbert_config = ColBERTConfig(
        doc_maxlen=128, 
        query_maxlen=512,
        bsize=4,
        lr=3e-6, 
        accumsteps=1,
        use_ib_negatives=False
    )
    
    colbert_config.checkpoint = args.colbert_model
    colbert_config.bsize = colbert_config.bsize // colbert_config.nranks

    doc_tokenizer = DocTokenizer(colbert_config)
    query_tokenizer = QueryTokenizer(colbert_config)

    colbert = ColBERT(name=colbert_config.checkpoint, colbert_config=colbert_config).cuda()

    doc_tokens = [doc_tokenizer.tensorize(x) for x in doc]

    query_tokens = [query_tokenizer.tensorize([example["input"]]) for example in list_data_dict]
    for i, example in enumerate(list_data_dict):
        doc_token = doc_tokens[i]
        query_token = query_tokens[i]
        Q = colbert.query(*query_token)
        D, D_mask = colbert.doc(*doc_token, keep_dims='return_mask')

        # Repeat each query encoding for every corresponding document.
        scores = colbert_score(Q, D, D_mask, config=colbert_config)
        
        selected_indices = torch.argsort(scores)[-5:]
        doc_select = [doc[i][j] for j in selected_indices]
        text = 'Reference:\n' + '\nReference:\n'.join(doc_select) + '\nQuestion:\n' + list_data_dict[i]['input'] + '\nPlease only give the correct option. Correct Answer:'
        if len(text.split()) > 5000:
            selected_indices = torch.argsort(scores)[-5:]
            selected_indices_revise = []
            for j in selected_indices:
                if len(doc[i][j].split()) > 1000:
                    continue
                selected_indices_revise.append(j)
            doc_select = [doc[i][j] for j in selected_indices_revise]
            text = 'Reference:\n' + '\nReference:\n'.join(doc_select) + '\nQuestion:\n' + list_data_dict[i]['input'] + '\nPlease only give the correct option. Correct Answer:'
        sources.append(text)

    
    #sources = ['Reference:\n' + '\nReference:\n'.join(example['doc'][:5]) + '\nQuestion:\n' + example['input'] + '\nPlease only give the correct option. Correct Answer:' for example in list_data_dict]
    #sources = [example['input'] + '\nPlease Only give the correct option. Correct Answer:' for example in list_data_dict]

    

    results = []
    outputs = [example['output'][0] for example in list_data_dict]
    def cal_acc(results, outputs):
        acc = 0
        result = []
        for i in range(len(results)):
            try:
                if results[i][0] == outputs[i]:
                    acc += 1
                else:
                    print(results[i][0], outputs[i])
                result.append(str(results[i][0]) + str(outputs[i]))
            except:
                continue
        print(acc / len(results))
    error_20 = []
    for i, x in enumerate(sources):
        output = respond(prompt=x)
        results.append(output)
        if results[i][0] != outputs[i]:
            error_20.append(list_data_dict[i])
    cal_acc(results, outputs)
    
    """
    with open(f'error_20.json', 'w') as file:
        json.dump(error_20, file, indent=4)
    import re
    outputs = []
    for example in list_data_dict:
        outputs.append(re.findall(r'Correct Answer: ([A-Z])' ,example['output'])[0])
    """
    
    ipdb.set_trace()
    
    with open(f'results/run_13b_amboss.txt', 'w') as file:
        file.write('\n'.join(results))
    
if __name__ == "__main__":
    args = parse_config()
    main(args)