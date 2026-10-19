#!/bin/bash
HEAD_IP=10.100.8.1

docker rm -f ray-head 2>/dev/null

docker run -d --name ray-head \
  --gpus all \
  --network host \
  --ipc=host \
  --shm-size=16G \
  --cap-add=IPC_LOCK \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e NCCL_SOCKET_IFNAME=enp1s0f1np1 \
  -e GLOO_SOCKET_IFNAME=enp1s0f1np1 \
  -e NCCL_IB_DISABLE=1 \
  --entrypoint bash \
  vllm-claude \
  -c "rm -rf /tmp/ray && ray start --head --port=6379 --node-ip-address=$HEAD_IP --num-gpus=1 --block"

docker logs -f ray-head
