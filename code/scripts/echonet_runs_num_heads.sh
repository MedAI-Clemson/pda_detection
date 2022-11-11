for numheads in 4 8 16 32 64 128 256 512 1024 2048 # 1 2 
do
    artifactfolder="/zfs/wficai/pda/model_run_artifacts/echonet_numheads_"${numheads}"/"
    mkdir -p $artifactfolder
    python train_echonet.py \
        configs/echonet/echonet_config.yaml \
        --artifact-folder ${artifactfolder} \
        --video-csv "/zfs/wficai/Data/echonet_data/train_val_test_1.0.csv" \
        --frame-csv "/zfs/wficai/Data/echonet_data/frames.csv" \
        --num-heads ${numheads} \
        --device 'cuda:1' \
        2>&1 | tee ${artifactfolder}log.txt
done