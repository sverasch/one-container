#!/bin/bash
WORKER_IP=$(ip -4 -o addr show | awk '{print $4}' | cut -d/ -f1 | grep '^10\.100\.' | head -n1)
[ -n "WORKER_IP" ] || { echo "No 10.100.x.x address found on this host"; exit 1; }
echo "Using worker IP $WORKER_IP"

docker rm -f vllm-worker 2>/dev/null

docker run -d --name vllm-worker \
  --gpus all \
  --network host \
  --shm-size=16G \
  --cap-add=IPC_LOCK \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HEAD_IP=10.0.1.176 \
  -e WORKER_IP=$WORKER_IP \
  -e NCCL_DEBUG=INFO \
  -e NCCL_SOCKET_IFNAME=enp1s0f1np1 \
  -e GLOO_SOCKET_IFNAME=enp1s0f1np1 \
  -e NCCL_IB_DISABLE=1 \
  -v $PWD/join-ray.sh:/join.sh:ro \
  --entrypoint bash \
  vllm-claude \
  /join.sh

