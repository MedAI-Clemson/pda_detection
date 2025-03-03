#!/bin/bash
#SBATCH --job-name=pda_ensemble
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=25gb
#SBATCH --gres=gpu:1
#SBATCH --partition=wficai
#SBATCH --output=output.log
#SBATCH --error=error.log
#SBATCH --time=4:00:00

echo "Running on host: "$HOSTNAME
echo "-------------------------------"
start=`date +%s`

# setup env
module load anaconda3/2022.10
module load cuda/11.7.0
source activate pda
cd /home/dane2/Code/pda_detection/code

# loop over number of heads
artifactfolder="/project/dane2/wficai/pda/model_run_artifacts/pda_ensemble/"
mkdir -p $artifactfolder
python train_pda_cv.py \
    configs/pda/pda_config.yaml \
    --artifact-folder ${artifactfolder} \
    --frame-csv "/project/dane2/wficai/pda/model_data/pda_train_val_test_remapped.csv" \
    --device 'cuda:0' \
    --pooling-method 'attn' \
    --num-heads 4 \
    --epochs 50 \
    2>&1 | tee ${artifactfolder}log.txt

end=`date +%s`
runtime=$((end-start))
echo "-------------------------------"
echo "Total runtime (s): "$runtime
echo "-------------------------------"
