for pool in "max" "avg"
do
    artifactfolder="/zfs/wficai/pda/model_run_artifacts/echonet_pool_"${pool}"/"
    mkdir -p $artifactfolder
    python train_echonet.py \
        configs/echonet/echonet_config.yaml \
        --artifact-folder ${artifactfolder} \
        --video-csv "/zfs/wficai/Data/echonet_data/train_val_test_1.0.csv" \
        --frame-csv "/zfs/wficai/Data/echonet_data/frames.csv" \
        --device 'cuda:1' \
        --pooling-method ${pool} \
        2>&1 | tee ${artifactfolder}log.txt
done