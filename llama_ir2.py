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
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from colbert.modeling.colbert import colbert_score
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer
from colbert.data import Queries
from colbert import Searcher
import csv
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import numpy as np
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




class LlamaIRModel(LlamaModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super(LlamaIRModel, self).__init__(config)
        self.colbert = None
        self.colbert_config = None

        # Initialize weights and apply final processing
        self.post_init()
        
    def initialize_colbert(self, colbert_config):
        self.colbert = ColBERT(name=colbert_config.checkpoint, colbert_config=colbert_config)
        self.colbert_config = colbert_config

    

    
class LlamaIRForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super(LlamaIRForCausalLM, self).__init__(config)
        self.model = LlamaIRModel(config)
        self.model.colbert_config = None
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        self.sigmoid = nn.Sigmoid()
        self.tokenizer = None
        
        self.post_init()

    def forward(self, 
        input_ids: torch.LongTensor = None,
        labels: Optional[torch.LongTensor] = None,
        doc_input_ids: Optional[List[torch.Tensor]] = None,
        doc_token: torch.LongTensor = None,
        query_token: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        
        Q = self.model.colbert.query(*query_token[0])
        D, D_mask = self.model.colbert.doc(*doc_token[0], keep_dims='return_mask')

        # Repeat each query encoding for every corresponding document.
        scores = colbert_score(Q, D, D_mask, config=self.model.colbert_config)
        normalized_scores = (scores - scores.min()) / (scores.max() - scores.min())
        scaled_scores = normalized_scores * 2 - 1

        probabilities = F.softmax(scaled_scores, dim=0)

        # Perform weighted random sampling
        selected_indices = torch.multinomial(probabilities, num_samples=10, replacement=False)

        

        loss_list = []
        labels = labels[0]
        
        with torch.no_grad():
            for i in selected_indices:
                doc = doc_input_ids[0][i]
                input_ids_i = torch.cat((doc, input_ids[0][1:]))
            

                label_i = torch.cat([torch.full((len(input_ids_i) - len(labels),), IGNORE_INDEX, dtype=labels.dtype, device=labels.device), labels])


                input_ids_i = torch.nn.utils.rnn.pad_sequence(
                    [input_ids_i], batch_first=True, padding_value=self.tokenizer.pad_token_id
                )
                label_i = torch.nn.utils.rnn.pad_sequence([label_i], batch_first=True, padding_value=IGNORE_INDEX)


                loss_i = super(LlamaIRForCausalLM, self).forward(
                    input_ids=input_ids_i.reshape(1, -1), 
                    labels=label_i.reshape(1, -1), 
                    attention_mask=input_ids_i.ne(self.tokenizer.pad_token_id), 
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds, use_cache=use_cache,
                    output_attentions=output_attentions, output_hidden_states=output_hidden_states,
                    return_dict=return_dict
                ).loss.detach()
                loss_list.append(loss_i)
        
        num = 1
        total_loss = torch.tensor(0.0).to(scaled_scores.device)
        for i in range(len(selected_indices)):
            for j in range(i+1, len(selected_indices)):
                pred_diff = scaled_scores[selected_indices[i]] - scaled_scores[selected_indices[j]]
                # 计算真实标签
                true_diff = loss_list[j] - loss_list[i]
                true_label = 1 if true_diff > 0 else 0
                # 计算单个成对的损失
                single_loss = -true_label * torch.log(self.sigmoid(pred_diff)) - (1 - true_label) * torch.log(1 - self.sigmoid(pred_diff))
                total_loss += single_loss
                num += 1
        
        selected_indices = torch.argsort(scores)[-5:]
        input_ids_summary = doc_input_ids[0][selected_indices[0]]
        for i in selected_indices[1:]:
            doc = doc_input_ids[0][i]
            input_ids_summary = torch.cat([input_ids_summary, doc[1:]])
        
        input_ids_summary = torch.cat([input_ids_summary, input_ids[0][1:]])

        label_summary = torch.cat([torch.full((len(input_ids_summary) - len(labels),), IGNORE_INDEX, dtype=labels.dtype, device=labels.device), labels])

        
        input_ids_summary = torch.nn.utils.rnn.pad_sequence(
            [input_ids_summary], batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        label_summary = torch.nn.utils.rnn.pad_sequence([label_summary], batch_first=True, padding_value=IGNORE_INDEX)

        output = super(LlamaIRForCausalLM, self).forward(
            input_ids=input_ids_summary, labels=label_summary, attention_mask=input_ids_summary.ne(self.tokenizer.pad_token_id), past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        if total_loss != 0:
            output.loss += total_loss / num
        return output

    
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, doc_input_ids=None, doc_token=None, query_token=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "doc_input_ids": doc_input_ids,
                "doc_token": doc_token,
                "query_token": query_token
            }
        )
        return model_inputs

    

