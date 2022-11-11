for num_heads in 1 2 4 8 16 32 64 128 256 512 1024 2048
do
    artifactfolder="/zfs/wficai/pda/model_run_artifacts/pda_scaled_dotprod_numheads_"${num_heads}"/"
    mkdir -p $artifactfolder
    python train_pda.py \
        configs/pda/pda_config.yaml \
        --artifact-folder ${artifactfolder} \
        --frame-csv /zfs/wficai/pda/model_data/pda_train_val_test.csv \
        --device 'cuda:1' \
        --num-heads ${num_heads} 2>&1 | tee ${artifactfolder}log.txt
done