#!/bin/bash
#PBS -N pda_numpatients
#PBS -l select=1:ncpus=56:mem=372gb:ngpus=1:gpu_model=a100,walltime=4:00:00
#PBS -q wficai
#PBS -j oe

echo "Running on host: "$HOSTNAME
echo "-------------------------------"
start=`date +%s`
# data_root_dir='/zfs/wficai/'
data_root_dir='/project/rcde/datasets/'

# setup env
module load anaconda3/2022.05-gcc/9.5.0
module load cuda/11.6.2-gcc/9.5.0
source activate pda
cd /home/dane2/Code/pda_detection/code

for subsetcol in "patient_sample_5" "patient_sample_12" "patient_sample_20" "patient_sample_28" "patient_sample_36" "patient_sample_44"
do
    artifactfolder=${data_root_dir}"pda/model_run_artifacts/MICCAI_pda_numpatients_"${subsetcol}"/"
    mkdir -p $artifactfolder
    python train_pda.py \
        configs/pda/pda_config.yaml \
        --artifact-folder ${artifactfolder} \
        --frame-csv ${data_root_dir}"/pda/model_data/pda_train_val_test.csv" \
        --num-heads 16 \
        --device 'cuda:0' \
        --subset-column ${subsetcol} 2>&1 | tee ${artifactfolder}log.txt
done

end=`date +%s`
runtime=$((end-start))
echo "-------------------------------"
echo "Total runtime (s): "$runtime
echo "-------------------------------"