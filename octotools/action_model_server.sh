export CUDA_VISIBLE_DEVICES=4,5,6,7
vllm serve ZhuofengLi/tool-n1-reason-lora-sft-800-step \
  -tp 4