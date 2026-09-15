#!/bin/bash

# List of parameters
n=1000
s_values=({1..10})
# lambda_values=(0.05)
K_values=10
# lambda_values=(0.01 0.02 0.05 0.1 0.2 0.5 1)
# lambda_values=1.0
lambda_values=(0.1) ## hp, 4week rnn
# lambda_values=(0.05) ## 4week, newdata
# lambda_values=(0.01) ## GBZ
# lambda_values=(0.5) ## GBZ 0.1s

## 4week data

# files=(
#   "4weekdata/p1a2_4week.npy"  "4weekdata/p1b2_4week.npy"  "4weekdata/p3a1_4week.npy"  "4weekdata/p3b1_4week.npy"  "4weekdata/p4b1_4week.npy"  "4weekdata/p4b3_4week.npy" 
#   "4weekdata/p5a2_4week.npy"  "4weekdata/p5b2_4week.npy"  "4weekdata/p1a3_4week.npy"  "4weekdata/p2a1_4week.npy"  "4weekdata/p3a2_4week.npy"  "4weekdata/p4a2_4week.npy"
#   "4weekdata/p4b2_4week.npy"  "4weekdata/p5a1_4week.npy"  "4weekdata/p5a3_4week.npy"
# )
# files=(
#   "4weekdata/p1a2_4week.npy"
# )
files=(
  "4weekdata/p1b2_4week.npy"  "4weekdata/p5a2_4week.npy"  "4weekdata/p3b1_4week.npy"
)
# files=("concatdata/MEC_220425_2A_week4.npy")

## HP find lambda
# files=("hp_files/tc050_DIV21.npy")

## HP recon error
# files=(
#   "hp_files/tc043_DIV14.npy"  "hp_files/tc165_DIV14.npy" "hp_files/tc165_DIV21.npy"
# )

# files=("hp_dup/tc043_DIV14_dup.npy" "hp_dup/tc043_tc165_DIV14.npy")
## HP results
# files=(
#   "hp_files/tc043_DIV14.npy"  "hp_files/tc043_DIV28.npy"  "hp_files/tc046_DIV14.npy"  "hp_files/tc046_DIV28.npy"  "hp_files/tc048_DIV21.npy"  "hp_files/tc049_DIV28.npy"  "hp_files/tc050_DIV21.npy"  "hp_files/tc051_DIV14.npy"  "hp_files/tc051_DIV28.npy"  "hp_files/tc164_DIV21.npy"  "hp_files/tc165_DIV14.npy"  "hp_files/tc165_DIV28.npy"
#   "hp_files/tc043_DIV21.npy"  "hp_files/tc045_DIV28.npy"  "hp_files/tc046_DIV21.npy"  "hp_files/tc048_DIV14.npy"  "hp_files/tc048_DIV28.npy"  "hp_files/tc050_DIV14.npy"  "hp_files/tc050_DIV28.npy"  "hp_files/tc051_DIV21.npy"  "hp_files/tc164_DIV14.npy"  "hp_files/tc164_DIV28.npy"  "hp_files/tc165_DIV21.npy"
# )
# files=("hp_files/tc043_DIV14.npy" "hp_files/tc165_DIV21.npy"  "hp_dup/tc043_DIV14_dup.npy"  "hp_dup/tc043_DIV14_tc164_DIV21.npy")

# files=("hp_dup/tc043_DIV14_dup.npy")
# files=(
# "hp_dup/tc043_DIV14_dup.npy"  "hp_dup/tc043_DIV14_tc048_DIV21.npy"  "hp_dup/tc043_DIV14_tc164_DIV21.npy"  "hp_dup/tc043_DIV14_tc164_DIV28.npy"  "hp_dup/tc043_tc165_DIV14.npy"  "hp_dup/tc048_tc164_DIV21.npy"
# )

## NEW data
# files=(
# "concatdata/MEC_220425_2A_week4.npy")
# files=(
# "concatdata/MEC_220425_6A_week4.npy"   "concatdata/MPT_220603_4B_week4.npy"
# "concatdata/MEC_220425_2B_week4.npy"  "concatdata/MEC_220425_6B_week4.npy"   "concatdata/MPT_220603_4C_week4.npy"
# "concatdata/MEC_220425_2C_week4.npy"  "concatdata/MEC_220425_6C_week4.npy"   "concatdata/MPT_220603_4D_week4.npy"
# "concatdata/MEC_220425_2D_week4.npy"  "concatdata/MEC_220425_6D_week4.npy"   "concatdata/MPT_220603_5A_week4.npy"
# "concatdata/MEC_220425_2E_week4.npy"  "concatdata/MEC_220425_6E_week4.npy"   
# "concatdata/MEC_220425_2F_week4.npy"  "concatdata/MEC_220425_7A_week4.npy"   "concatdata/MPT_220603_5C_week4.npy"
# "concatdata/MEC_220425_3B_week4.npy"  "concatdata/MEC_220425_7B_week4.npy"   "concatdata/MPT_220603_6C_week4.npy"
# "concatdata/MEC_220425_3C_week4.npy"  "concatdata/MPT_220603_7B_week4.npy"
# "concatdata/MEC_220425_3E_week4.npy"  "concatdata/MPT_220603_10A_week4.npy"  "concatdata/MPT_220603_7D_week4.npy"
# "concatdata/MEC_220425_3F_week4.npy"  "concatdata/MPT_220603_10D_week4.npy"  "concatdata/MPT_220603_8A_week4.npy"
# "concatdata/MEC_220425_4A_week4.npy"  "concatdata/MPT_220603_1A_week4.npy"   "concatdata/MPT_220603_8B_week4.npy"
# "concatdata/MEC_220425_4B_week4.npy"  "concatdata/MPT_220603_1B_week4.npy"   "concatdata/MPT_220603_8C_week4.npy"
# "concatdata/MEC_220425_4C_week4.npy"  "concatdata/MPT_220603_1C_week4.npy"   "concatdata/MPT_220603_9B_week4.npy"
# "concatdata/MEC_220425_4D_week4.npy"  "concatdata/MPT_220603_2C_week4.npy"   "concatdata/MPT_220603_9C_week4.npy"
# "concatdata/MEC_220425_4E_week4.npy"  "concatdata/MPT_220603_3A_week4.npy"   
# "concatdata/MEC_220425_4F_week4.npy"  "concatdata/MPT_220603_3B_week4.npy"   
# "concatdata/MEC_220425_5B_week4.npy"  "concatdata/MPT_220603_3C_week4.npy"   "concatdata/MPT_220606_5B_week4.npy"
# "concatdata/MEC_220425_5C_week4.npy"  "concatdata/MPT_220603_4A_week4.npy"   
# )

## GBZ data

# files=("GBZ250/230330_MOS_ALICO1_concatenated.npy")

# files=(
# "GBZ250/230330_MOS_ALICO1_concatenated.npy"  "GBZ250/230330_MT_ALICO6_concatenated.npy"  "GBZ250/230331_WT_ALICO_concatenated.npy"    "GBZ250/240705_WT_ALICO2_concatenated.npy"  "GBZ250/240709_MT_ALICO3_concatenated.npy"  "GBZ250/240709_WT_ALICO4_concatenated.npy"  "GBZ250/240712_WT_ALICO5_concatenated.npy"
# "GBZ250/230330_MOS_ALICO6_concatenated.npy"  "GBZ250/230330_WT_ALICO7_concatenated.npy"  "GBZ250/240705_MOS_ALICO2_concatenated.npy"  "GBZ250/240705_WT_ALICO3_concatenated.npy"  "GBZ250/240709_WT_ALICO1_concatenated.npy"  "GBZ250/240712_MT_ALICO1_concatenated.npy"
# "GBZ250/230330_MT_ALICO2_concatenated.npy"   "GBZ250/230331_MT_ALICO1_concatenated.npy"  "GBZ250/240705_MT_ALICO5_concatenated.npy"   "GBZ250/240709_MT_ALICO2_concatenated.npy"  "GBZ250/240709_WT_ALICO3_concatenated.npy"  "GBZ250/240712_MT_ALICO7_concatenated.npy"
# )

## GBZ data 0.1s
# files=("GBZ250_01/230330_MOS_ALICO1_concatenate01.npy")
# files=(
# "GBZ250_01/230330_MOS_ALICO1_concatenate01.npy"  "GBZ250_01/230330_MT_ALICO6_concatenate01.npy"  "GBZ250_01/230331_WT_ALICO_concatenate01.npy"    "GBZ250_01/240705_WT_ALICO2_concatenate01.npy"  "GBZ250_01/240709_MT_ALICO3_concatenate01.npy"  "GBZ250_01/240709_WT_ALICO4_concatenate01.npy"  "GBZ250_01/240712_WT_ALICO5_concatenate01.npy"
# "GBZ250_01/230330_MOS_ALICO6_concatenate01.npy"  "GBZ250_01/230330_WT_ALICO7_concatenate01.npy"  "GBZ250_01/240705_MOS_ALICO2_concatenate01.npy"  "GBZ250_01/240705_WT_ALICO3_concatenate01.npy"  "GBZ250_01/240709_WT_ALICO1_concatenate01.npy"  "GBZ250_01/240712_MT_ALICO1_concatenate01.npy"
# "GBZ250_01/230330_MT_ALICO2_concatenate01.npy"   "GBZ250_01/230331_MT_ALICO1_concatenate01.npy"  "GBZ250_01/240705_MT_ALICO5_concatenate01.npy"   "GBZ250_01/240709_MT_ALICO2_concatenate01.npy"  "GBZ250_01/240709_WT_ALICO3_concatenate01.npy"  "GBZ250_01/240712_MT_ALICO7_concatenate01.npy"
# )


job_count=0
max_jobs=200
# Loop over each combination of parameters
for s in "${s_values[@]}"; do
  for K in "${K_values[@]}"; do
    for lambda_value in "${lambda_values[@]}"; do
      echo "Running combination -s $s -K $K on all files..."

    # Run each file in parallel for this combination of s and K
      for file in "${files[@]}"; do
        device=$((job_count % 4))
        
        # Extract the base name of the file without the directory
        base_file=$(basename "$file")
        
        echo "Running with parameters: -n $n -s $s -lambda_h $lambda_value -K $K -file $file -device $device"
        python3 test_realdata.py -n $n -s $s -lambda_h $lambda_value -K $K -file $file -device $device &
        job_id=$!
        echo "Started job with ID: $job_id"
        job_count=$((job_count + 1))
      if [ "$job_count" -ge "$max_jobs" ]; then
        echo "Waiting for the current batch of jobs to finish..."
        wait  # Wait for all background jobs to finish
        job_count=0  # Reset the job counter
      fi
      done
      
      # Wait for all jobs (for this specific combination of s and K) to complete
      echo "Waiting for jobs with s=$s and K=$K to finish..."
      # wait  # Wait for all background jobs to finish before moving to the next s and K combination
    done
  done
done

# Final message when all combinations are done
echo "All jobs for all s and K combinations have been completed."