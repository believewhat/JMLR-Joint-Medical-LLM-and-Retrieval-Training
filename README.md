# JMLR-Joint-Medical-LLM-and-Retrieval-Training

Large Language Models (LLMs) have demonstrated a remarkable potential in medical knowledge acquisition and question-answering. However, LLMs can potentially hallucinate and yield factually incorrect outcomes, even with domain-specific pretraining. Previously, retrieval augmented generation (RAG) has limited success in addressing hallucinations. Unlike previous methods in RAG where the retrieval model was trained separately from the LLM, we introduce JMLR (for Jointly trains LLM and information Retrieval (IR)) during the fine-tuning phase. The synchronized training mechanism enhances JMLR's ability to retrieve clinical guidelines and leverage medical knowledge to reason and answer questions and reduces the demand for computational resources. We evaluated JMLR on the important medical question answering application. Our experimental results demonstrate that JMLR-13B (70.5%) outperforms a previous state-of-the-art open-source model using conventional pre-training and fine-tuning Meditron-70B (68.9%) and Llama2-13B with RAG (54.9%) on a medical question-answering dataset. JMLR-13B (148 GPU hours) also trains much faster than Meditron-70B (42630 GPU hours). Through this work, we provide a new and efficient knowledge enhancement tool for healthcare, demonstrating the potential of integrating IR and LLM training for medical question-answering systems. The code, along with selected retrieval data that can be made public, is included in the supplementary material and will be made publicly accessible with CC-BY 4.0 license upon the paper's acceptance.



![Image text](https://github.com/believewhat/JMLR-Joint-Medical-LLM-and-Retrieval-Training/blob/main/figure/sample_figure.png)

**DataSet**

We utilized GPT-4 to generate corresponding reasoning for each question on MedQA, and these reasonings were then incorporated into the guidelines. You could download them from [MedQA-Reason](https://huggingface.co/datasets/akemiH/MedQA-Reason)


**How to use**

We release our model for MedQA. You could download the model from [JMLR-13B](https://huggingface.co/akemiH/JMLR)

```python
CUDA_VISIBLE_DEVICES=0 python inference_ir.py \
    --base_model  Your Path \
    --data_path Your Path \
    --context_size 32768 \
    --colbert_model  Your Path\
    --max_gen_len 10
```


