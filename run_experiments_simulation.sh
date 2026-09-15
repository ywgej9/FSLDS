#!/bin/bash

n=1000
s_values=({1..35})
# lambda_values=(0.05 0.2 0.5 1.0)
lambda_values=(1.0)
K_values=(6)

num_gpus=4
max_jobs=20       # total concurrent jobs
active_jobs=0
launch_count=0

for K in "${K_values[@]}"; do
  for s in "${s_values[@]}"; do
    for lambda_value in "${lambda_values[@]}"; do

      # If max_jobs are already running, wait until ONE finishes
      if [ "$active_jobs" -ge "$max_jobs" ]; then
        wait -n
        active_jobs=$((active_jobs - 1))
      fi

      # Round-robin assignment across GPUs
      device=$((launch_count % num_gpus))

      echo "Running: K=$K seed=$s lambda=$lambda_value device=$device"

      python3 test_simulation.py \
        -n "$n" \
        -s "$s" \
        -K "$K" \
        -lambda_h "$lambda_value" \
        -device "$device" &

      active_jobs=$((active_jobs + 1))
      launch_count=$((launch_count + 1))

    done
  done
done

# Wait for remaining jobs
wait

echo "All jobs finished."