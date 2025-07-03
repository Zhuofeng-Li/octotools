export CUDA_VISIBLE_DEVICES=4,5,6,7
vllm serve Qwen/Qwen2.5-7B-Instruct \
  -tp 4