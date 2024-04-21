#!/bin/sh
#
# nohup bash checkpubmedgpt.sh > checkpubmedgpt.out 2> checkpubmedgpt.err &
# baseline model 70371
# --model_name_or_path /data/data_user_omega/llm_share/flant5/flan-t5-xl \

#cd /home/zhichaoyang/pubmedgpt_ct_data/llama/viscuna
sleep 1
export CUDA_DEVICE_ORDER="PCI_BUS_ID" 

export CUDA_HOME="/usr/local/cuda-12.2"
export LD_LIBRARY_PATH="/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH"
export PATH="/usr/local/cuda-12.2/bin:$PATH"


# llama2-7b bioinstruct
# {'mmlu_loss': 1.255663034877157, 'mmlu_eval_accuracy_anatomy': 0.6428571428571429, 'mmlu_eval_accuracy_medical_genetics': 0.8181818181818182, 'mmlu_eval_accuracy_professional_medicine': 0.3225806451612903, 'mmlu_eval_accuracy_college_medicine': 0.3181818181818182, 'mmlu_eval_accuracy_clinical_knowledge': 0.4827586206896552, 'mmlu_eval_accuracy_college_biology': 0.5625, 'mmlu_eval_accuracy': 0.5245100075119541}
# ***** eval metrics *****
#   mmlu_eval_accuracy                       = 0.5245100075119541
#   mmlu_eval_accuracy_anatomy               =             0.6429
#   mmlu_eval_accuracy_clinical_knowledge    =             0.4828
#   mmlu_eval_accuracy_college_biology       =             0.5625
#   mmlu_eval_accuracy_college_medicine      =             0.3182
#   mmlu_eval_accuracy_medical_genetics      =             0.8182
#   mmlu_eval_accuracy_professional_medicine =             0.3226
#   mmlu_loss                                =             1.2557


#/home/jwang/Project/doctorrobot/LongLoRA/colbert/colbertv2.0
#/data/experiment_data/junda/chatdoctor/llama-13b-32k-usmle-open-ir/checkpoint-1779/ir
CUDA_VISIBLE_DEVICES=0 python inference_ir.py \
    --base_model  /data/experiment_data/junda/chatdoctor/llama-13b-32k-medqa-open-reason-hf-2 \
    --data_path /home/jwang/Project/doctorrobot/LongLoRA/usmle_open_test.json \
    --context_size 32768 \
    --colbert_model  /data/experiment_data/junda/chatdoctor/llama-13b-32k-medqa-open-ir/checkpoint-1/ir\
    --max_gen_len 10

