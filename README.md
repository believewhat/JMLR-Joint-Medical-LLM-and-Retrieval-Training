# JMLR-Joint-Medical-LLM-and-Retrieval-Training

Both our training process and the conventional finetuning approach employ the AdamW optimizer, with $\beta_1 = 0.9$, $\beta_2 = 0.95$, and $\text{eps} = 1 \times 10^{-5}$. We implement a cosine learning rate schedule, incorporating a warmup phase that accounts for 10% of the training duration and decays the learning rate to 10% of its peak value. In alignment with the practices outlined in Llama 2-chat \cite{Touvron2023Llama2O}, our training employs a learning rate of $1 \times 10^{-5}$, a weight decay factor of 0.1, and manages a batch size of 2. The finetuning phase spans 5 epochs for all iterations. However, we apply a distinct learning rate for ColBERT, set at $3 \times 10^{-5}$. The optimization strategy for training ColBERT mirrors that used for Llama 2. Throughout these experiments, we utilize four A100 80 GB GPUs. The total training duration for our method amounts to 25 hours. Additionally, the pretraining phase on medical guidelines takes approximately 138 hours, while the finetuning phase is approximately 17 hours on 7B Llama. This comprehensive training approach ensures the effective adaptation and optimization of the model for specific medical applications.

![Image text](https://github.com/believewhat/JMLR-Joint-Medical-LLM-and-Retrieval-Training/blob/main/figure/sample_figure.png)

**How to use**

You could download our model from [Huggingface]([https://drive.google.com/file/d/1wwXYF9ictgZQ0DyxRsbkP5M6tXHxExsC/view?usp=sharing](https://huggingface.co/akemiH/JMLR))

```python
CUDA_VISIBLE_DEVICES=0 python inference_ir.py \
    --base_model  Your Path \
    --data_path Your Path \
    --context_size 32768 \
    --colbert_model  Your Path\
    --max_gen_len 10
```
