#!/bin/bash
#PBS -N echonet_numvid
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

for dataset in \
    "train_fullval_test_0.01.csv"  "train_fullval_test_0.05.csv" "train_fullval_test_0.1.csv" \
    "train_fullval_test_0.25.csv" "train_fullval_test_0.5.csv" "train_fullval_test_0.75.csv" \
    "train_val_test_1.0.csv"
do
    artifactfolder="/zfs/wficai/pda/model_run_artifacts/MICCAI_echonet_numvids_"${dataset}"/"
    datapath="/zfs/wficai/Data/echonet_data/"${dataset}"/"
    mkdir -p $artifactfolder
    python train_echonet.py \
        configs/echonet/echonet_config.yaml \
        --artifact-folder ${artifactfolder} \
        --video-csv "/zfs/wficai/Data/echonet_data/"${dataset} \
        --frame-csv "/zfs/wficai/Data/echonet_data/frames.csv" \
        --num-heads 32 \
        --device 'cuda:0' \
        2>&1 | tee ${artifactfolder}log.txt
done

end=`date +%s`
runtime=$((end-start))
echo "-------------------------------"
echo "Total runtime (s): "$runtime
echo "-------------------------------"