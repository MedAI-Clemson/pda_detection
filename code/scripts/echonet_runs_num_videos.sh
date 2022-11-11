for dataset in \
    "train_fullval_test_0.01.csv"  "train_fullval_test_0.05.csv" "train_fullval_test_0.1.csv" \
    "train_fullval_test_0.25.csv" "train_fullval_test_0.5.csv" "train_fullval_test_0.75.csv" \
    "train_val_test_1.0.csv"
do
    artifactfolder="/zfs/wficai/pda/model_run_artifacts/echonet_nh8_"${dataset}"/"
    datapath="/zfs/wficai/Data/echonet_data/"${dataset}"/"
    mkdir -p $artifactfolder
    python train_echonet.py \
        configs/echonet/echonet_config.yaml \
        --artifact-folder ${artifactfolder} \
        --video-csv "/zfs/wficai/Data/echonet_data/"${dataset} \
        --frame-csv "/zfs/wficai/Data/echonet_data/frames.csv" \
        --num-heads 8 \
        --device 'cuda:1' \
        2>&1 | tee ${artifactfolder}log.txt
done