import pandas as pd
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
import timm
from timm import optim
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn import metrics as skmet
import os
import yaml

# locally defined
import transforms as my_transforms
from dataset import PdaVideos, collate_video
import models
from utils import num_parameters, EarlyStopper

#set rng seeds
torch.manual_seed(2)
np.random.seed(2)

def train_one_epoch(model, optimizer, train_dataloader, device):
    model.train()

    num_steps_per_epoch = len(train_dataloader)

    losses = []
    for ix, batch in enumerate(train_dataloader):
        inputs = batch['video'].to(device)
        num_frames = batch['num_frames']
        targets = batch['trg_type'].to(device).type(torch.float32)
        outputs, _ = model(inputs, num_frames)
        loss = nn.functional.binary_cross_entropy_with_logits(outputs.squeeze(), targets.squeeze())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss.detach().item())
        print(f"\tBatch {ix+1} of {num_steps_per_epoch}. Loss={loss.detach().item():0.3f}", end='\r')
    
    print(' '*100, end='\r')
        
    return np.mean(losses)  
            
def evaluate(model, test_dataloader, device):
    model.eval()

    num_steps_per_epoch = len(test_dataloader)

    target_ls = []
    output_ls = []
    losses = []
    for ix, batch in enumerate(test_dataloader):
        inputs = batch['video'].to(device)
        num_frames = batch['num_frames']
        targets = batch['trg_type'].to(device).type(torch.float32)
        target_ls.append(targets.cpu().numpy())
        
        with torch.no_grad():
            outputs, _ = model(inputs, num_frames)
            output_ls.append(outputs.cpu().numpy())
            loss = nn.functional.binary_cross_entropy_with_logits(outputs.squeeze(), targets.squeeze())
            
        losses.append(loss.detach().item())
        
    metrics = compute_metrics(np.concatenate(target_ls), np.concatenate(output_ls))
    return np.mean(losses), metrics

def compute_metrics(y_true, y_pred):
    mets = dict()
    
    y_pred = 1/(1+np.exp(-y_pred))
    y_pred_cls = (y_pred>0.5).astype(int)
    
    mets['roc_auc'] = skmet.roc_auc_score(y_true, y_pred)
    mets['accuracy'] = skmet.accuracy_score(y_true, y_pred_cls)
    mets['sensitivity'] = skmet.recall_score(y_true, y_pred_cls)
    mets['specificity'] = skmet.recall_score(y_true, y_pred_cls, pos_label=0)
    
    return mets

def main(cfg):
    os.makedirs(cfg['artifact_folder'], exist_ok=True)

    # copy the config file to the artifact folder
    with open(cfg['artifact_folder'] + '/config.yaml', 'w') as f: 
        yaml.dump(cfg, f)
    
    device = torch.device(cfg['device'])
    
    # classifier network
    print("Creating video classifier network.")
    encoder = timm.create_model(**cfg['encoder_kwargs'], num_classes=0)
    m = models.MedVidNet(encoder, **cfg['vidnet_kwargs']).to(device)
    
    num_pars = num_parameters(m)
    num_pars_encoder = num_parameters(encoder)
    print(f"Number of trainable params: {num_pars} ({num_pars - num_pars_encoder} excluding encoder).")

    # transforms
    tfms = my_transforms.VideoTransforms(cfg['res'], time_downsample_kwargs = cfg['time_downsample_kwargs'])
    tfms_train = tfms.get_transforms('train')
    tfms_test = tfms.get_transforms('test')

    # load data
    print("Preparing datasets.")
    df_frames = pd.read_csv(cfg['frame_csv'])

    # create datasets
    d_train = PdaVideos(df_frames, transforms = tfms_train, split='TRAIN', subset_column=cfg['subset_column'], **cfg['dataset_kwargs'])
    dl_train = DataLoader(d_train, shuffle=True, collate_fn=collate_video, **cfg['dataloader_kwargs'])
    d_val = PdaVideos(df_frames, transforms = tfms_test, split='VAL', **cfg['dataset_kwargs'])
    dl_val= DataLoader(d_val, collate_fn=collate_video, **cfg['dataloader_kwargs'])
    d_test = PdaVideos(df_frames, transforms = tfms_test, split='TEST', **cfg['dataset_kwargs'])
    dl_test= DataLoader(d_test, collate_fn=collate_video, **cfg['dataloader_kwargs'])

    print("Train data size:", len(d_train))
    print("Validation data size:", len(d_val))
    print("Test data size:", len(d_test))    

    print("Beginning training loop.")
    optimizer = optim.AdamP(m.parameters(), **cfg['optim_kwargs'])
    scheduler = ReduceLROnPlateau(optimizer, **cfg['sched_kwargs'])
    stopper = EarlyStopper(**cfg['early_stopper_kwargs'])

    train_loss_ls = []
    val_loss_ls = []
    metrics_ls = []
    best_val_loss = 1e10
    for epoch in range(cfg['num_epochs']):

        # train for a single epoch
        train_loss = train_one_epoch(m, optimizer, dl_train, device)
        train_loss_ls.append(train_loss)
        print(f"[{epoch} TRAIN] Cross entropy loss = {train_loss:0.5f}")       

        # evaluate
        val_loss, metrics = evaluate(m, dl_val, device)
        val_loss_ls.append(val_loss)
        print(f"[{epoch} VALID] Cross entropy loss = {val_loss:0.5f}")
        print(f"[{epoch} VALID] PDA classification: ", *[f"{k}={v:0.5f}" for k, v in metrics.items()])

        if val_loss < best_val_loss:
            print(f"Validation loss improved ({best_val_loss:0.5f} --> {val_loss:0.5f}). Saving model checkpoint.")
            torch.save(m.state_dict(), f"{cfg['artifact_folder']}/model_checkpoint_video.ckpt")
            best_val_loss =  val_loss 

        if stopper.stop(val_loss):
            print(f"Stopping early because validation loss did not improve after {stopper.patience} epochs.")
            break
            
        scheduler.step(val_loss)
        
    print("Post-training evaluation:")
    m.load_state_dict(torch.load(cfg['artifact_folder'] + 'model_checkpoint_video.ckpt'))
    print("Validation:")
    val_loss, metrics = evaluate(m, dl_val, device)
    print(f"\tCross entropy loss = {val_loss:0.5f}")
    print("\tPDA classification: ", *[f"{k}={v:0.5f}" for k, v in metrics.items()])
    
    print("Test:")
    test_loss, metrics = evaluate(m, dl_test, device)
    print(f"\tCross entropy loss = {test_loss:0.5f}")
    print("\tPDA classification: ", *[f"{k}={v:0.5f}" for k, v in metrics.items()])
        
if __name__=='__main__':
    import argparse
    import sys
    
    parser = argparse.ArgumentParser('Train pda video classifier')
    parser.add_argument('config', type=str, metavar='FILE',
                        help='YAML configuration file.')
    parser.add_argument('--artifact-folder', type=str, metavar='DIR', default=None,
                        help='Directory for model outputs. Overrides "artifact_folder" in config if provided.')
    parser.add_argument('--frame-csv', type=str, metavar='FILE', default=None,
                        help='CSV frame data file. Overrides "frame_csv" in config if provided.')
    parser.add_argument('--subset-column', type=str, required=False, default=None,
                        help='Name of boolean column in frames file to subset training data. Overrides "subset_column" in config if provided.')
    parser.add_argument('--num-heads', type=int, required=False, default=None,
                        help='Number of attention heads for MedVidNet model. Overrides "vidnet_kwargs: num_heads" in config if provided.')
    parser.add_argument('--pooling-method', type=str, required=False, default=None,
                        choices=['attn', 'max', 'avg'],
                        help='Frame pooling method. Overrides "vidnet_kwargs: pooling_method" in config if provided.')
    parser.add_argument('--device', type=str, required=False, default=None,
                        help='Compute devie. Overrides "device" in config if provided')
    args = parser.parse_args()
    
    # load config
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
        
    # override with command line args
    if args.artifact_folder is not None:
        cfg['artifact_folder'] = args.artifact_folder
    if args.frame_csv is not None:
        cfg['frame_csv'] = args.frame_csv
    if args.subset_column is not None:
        cfg['subset_column'] = args.subset_column
    if args.num_heads is not None:
        cfg['vidnet_kwargs']['num_heads'] = args.num_heads
    if args.pooling_method is not None:
        cfg['vidnet_kwargs']['pooling_method'] = args.pooling_method
    if args.device is not None:
        cfg['device'] = args.device

    cfg['artifact_folder'] = path...
    
    print("Running training script with configuration:")
    print('-'*30)
    yaml.dump(cfg, sys.stdout)
    print('-'*30)

    main(cfg)