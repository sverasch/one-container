#!/bin/bash
set -e

HEAD_IP=10.100.8.1
MODEL=Qwen/Qwen3.8-Flash-Next-FP8
EXPECTED_GPUS=2
TIMEOUT_SECS=600
LOG=/tmp/vllm.log


# Stop any vLLM engine left over from a previous launch before starting a new one
docker exec ray-head pkill -f "vllm serve" 2>/dev/null || true
docker exec ray-head pkill -f EngineCore 2>/dev/null || true

# 4. Launch vLLM in the background inside ray-head, logging to a file
docker exec -d \
  -e HF_HUB_OFFLINE=1 \
  -e VLLM_USE_FLASHINFER_SAMPLER=0 \
  -e NCCL_DEBUG=INFO \
  -e NCCL_SOCKET_IFNAME=enp1s0f1np1 \
  -e GLOO_SOCKET_IFNAME=enp1s0f1np1 \
  -e NCCL_IB_DISABLE=1 \
  -e RAY_ADDRESS=$HEAD_IP:6379 \
  -e VLLM_HOST_IP=$HEAD_IP \
  ray-head \
  bash -c "vllm serve $MODEL \
    --host 0.0.0.0 \
    --port 8000 \
    --max-num-seqs 256 \
    --distributed-executor-backend ray \
    --enable-prefix-caching \
    --enable-expert-parallel \
    --tensor-parallel-size 2 \
    --moe-backend triton \
    --gpu-memory-utilization 0.85 \
    --kv-cache-dtype fp8 \
    --attention-config.indexer_kv_dtype fp8 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":3}' \
    --limit-mm-per-prompt '{\"image\":5,\"video\":5}' \
    > $LOG 2>&1"

docker exec ray-head tail -f /tmp/vllm.log
echo "vLLM launched. Following $LOG (Ctrl+C stops the log, not vLLM)."
