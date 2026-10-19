#!/bin/bash
set -e

HEAD_IP=${HEAD_IP:?HEAD_IP must be set}
WORKER_IP=${WORKER_IP:?WORKER_IP must be set}
RAY_PORT=6379

rm -rf /tmp/ray

trap 'ray stop --force >/dev/null 2>&1' EXIT

ray stop --force >/dev/null 2>&1 || true
rm -rf /tmp/ray

ray start --address=$HEAD_IP:$RAY_PORT --node-ip-address=$WORKER_IP --num-gpus=$(nvidia-smi -L | wc -l)

echo "Joined Ray head at $HEAD_IP:$RAY_PORT"
sleep infinity
