#!/bin/bash
#PBS -N echonet_numheads
#PBS -l select=1:ncpus=64:mem=250gb:ngpus=1:gpu_model=a100:phase=29,walltime=14:00:00
#PBS -j oe

echo "Running on host: "$HOSTNAME
echo "-------------------------------"
start=`date +%s`

# setup env
module load anaconda3/2022.05-gcc/9.5.0
module load cuda/11.6.2-gcc/9.5.0
source activate pda
cd /home/dane2/Code/pda_detection/code

for numheads in 1 2 4 8 16 32 64 128 256 512 1024 2048
do
    artifactfolder="/zfs/wficai/pda/model_run_artifacts/MICCAI_echonet_numheads_"${numheads}"/"
    mkdir -p $artifactfolder
    python train_echonet.py \
        configs/echonet/echonet_config.yaml \
        --artifact-folder ${artifactfolder} \
        --video-csv "/zfs/wficai/Data/echonet_data/train_val_test_1.0.csv" \
        --frame-csv "/zfs/wficai/Data/echonet_data/frames.csv" \
        --num-heads ${numheads} \
        --pooling-method 'attn' \
        --device 'cuda:0' \
        2>&1 | tee ${artifactfolder}log.txt
done

end=`date +%s`
runtime=$((end-start))
echo "-------------------------------"
echo "Total runtime (s): "$runtime
echo "-------------------------------"