#!/bin/bash
#PBS -N pda_numheads
#PBS -l select=1:ncpus=56:mem=372gb:ngpus=1:gpu_model=a100:phase=27,walltime=4:00:00
#PBS -q wficai
#PBS -j oe

echo "Running on host: "$HOSTNAME
echo "-------------------------------"
start=`date +%s`

# setup env
module load anaconda3/2022.05-gcc/9.5.0
module load cuda/11.6.2-gcc/9.5.0
source activate pda
cd /home/dane2/Code/pda_detection/code

# loop over number of heads
for num_heads in 1 2 4 8 16 32 64 128 256 512 1024 2048
do
    artifactfolder="/project/rcde/datasets/pda/model_run_artifacts/MICCAI_pda_numheads_"${num_heads}"/"
    mkdir -p $artifactfolder
    python train_pda.py \
        configs/pda/pda_config_numheads.yaml \
        --artifact-folder ${artifactfolder} \
        --frame-csv "/project/rcde/datasets/pda/model_data/pda_train_val_test.csv" \
        --device 'cuda:0' \
        --pooling-method 'attn' \
        --num-heads ${num_heads} \
        2>&1 | tee ${artifactfolder}log.txt
done

end=`date +%s`
runtime=$((end-start))
echo "-------------------------------"
echo "Total runtime (s): "$runtime
echo "-------------------------------"
