for subsetcol in "patient_sample_5" "patient_sample_12" "patient_sample_20" "patient_sample_28" "patient_sample_36" "patient_sample_44"
do
    artifactfolder="/zfs/wficai/pda/model_run_artifacts/pda_"${subsetcol}"_maxpool/"
    mkdir -p $artifactfolder
    python train_pda.py \
        configs/pda/pda_config.yaml \
        --artifact-folder ${artifactfolder} \
        --frame-csv /zfs/wficai/pda/model_data/pda_train_val_test.csv \
        --num-heads 4 \
        --device 'cuda:0' \
        --pooling-method 'max' \
        --subset-column ${subsetcol} 2>&1 | tee ${artifactfolder}log.txt
done