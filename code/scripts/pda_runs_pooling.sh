for pool in "max" "avg"
do
    artifactfolder="/zfs/wficai/pda/model_run_artifacts/pda_"${pool}"/"
    mkdir -p $artifactfolder
    python train_pda.py \
        configs/pda/pda_config.yaml \
        --artifact-folder ${artifactfolder} \
        --frame-csv /zfs/wficai/pda/model_data/pda_train_val_test.csv \
        --device 'cuda:0' \
        --pooling-method ${pool} 2>&1 | tee ${artifactfolder}log.txt
done