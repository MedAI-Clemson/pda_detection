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
from dataset import EchoNetVideos, collate_video
import models
from utils import num_parameters, EarlyStopper

def train_one_epoch(model, optimizer, train_dataloader, device):
    model.train()

    num_steps_per_epoch = len(train_dataloader)

    losses = []
    for ix, batch in enumerate(train_dataloader):
        inputs = batch['video'].to(device)
        esv_true = batch['ESV'].to(device).type(torch.float32)
        edv_true = batch['EDV'].to(device).type(torch.float32)
        ef_true = 1 - esv_true / edv_true
        num_frames = batch['num_frames']
        out, at = model(inputs, num_frames)
        ef_pred = 1 - out[:,1]/out[:,0]
        
        # Loss
        loss = nn.functional.mse_loss(ef_pred.squeeze(), ef_true.squeeze())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss.detach().item())
        print(f"\tBatch {ix+1} of {num_steps_per_epoch}. Loss={loss.detach().item():0.5f}", end='\r')
    
    print(' '*100, end='\r')
        
    return np.mean(losses)  

def evaluate(model, val_dataloader, device):
    model.eval()

    num_steps_per_epoch = len(val_dataloader)

    ef_true_ls = []
    ef_pred_ls = []
    losses = []
    for ix, batch in enumerate(val_dataloader):
        inputs = batch['video'].to(device)
        esv_true = batch['ESV'].to(device).type(torch.float32)
        edv_true = batch['EDV'].to(device).type(torch.float32)
        ef_true = 1 - esv_true / edv_true
        ef_true_ls.append(ef_true.cpu().numpy())
        
        num_frames = batch['num_frames']
        
        with torch.no_grad():
            out, _ = model(inputs, num_frames)
            ef_pred = 1 - out[:,1]/out[:,0]
        
        ef_pred_ls.append(ef_pred.cpu().numpy())

        # Loss
        loss = nn.functional.mse_loss(ef_pred.squeeze(), ef_true.squeeze())
        losses.append(loss.detach().item())
        
    ef_true_ar = np.concatenate(ef_true_ls)
    ef_pred_ar = np.concatenate(ef_pred_ls)
    
    metrics_ef = compute_metrics(ef_true_ar, ef_pred_ar)
    
    return np.mean(losses), metrics_ef

def compute_metrics(y_true, y_pred):
    mets = dict()
    
    mets['r2'] = skmet.r2_score(y_true, y_pred)
    mets['mae'] = skmet.mean_absolute_error(y_true, y_pred)
    mets['rmse'] = np.sqrt(skmet.mean_squared_error(y_true, y_pred))
    
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
    
    # initialize output bias to reasonable values for
    # edv and esv
    m.fc_out[-1].bias.data[0] = 0.9
    m.fc_out[-1].bias.data[1] = 0.45
    
    num_pars = num_parameters(m)
    num_pars_encoder = num_parameters(encoder)
    print(f"Number of trainable params: {num_pars} ({num_pars - num_pars_encoder} excluding encoder).")

    # transforms
    tfms = my_transforms.VideoTransforms(cfg['res'], time_downsample_kwargs = cfg['time_downsample_kwargs'])
    tfms_train = tfms.get_transforms('train')
    tfms_test = tfms.get_transforms('test')

    # load data
    print("Preparing datasets.")
    df_videos = pd.read_csv(cfg['video_csv'])
    df_frames = pd.read_csv(cfg['frame_csv'])

    # create datasets
    d_train = EchoNetVideos(df_videos, df_frames, transforms = tfms_train, 
                            split='TRAIN')
    dl_train = DataLoader(d_train, shuffle=True, collate_fn=collate_video, **cfg['dataloader_kwargs'])
    d_val = EchoNetVideos(df_videos, df_frames, transforms = tfms_test,
                         split='VAL')
    dl_val= DataLoader(d_val, collate_fn=collate_video, **cfg['dataloader_kwargs'])
    d_test = EchoNetVideos(df_videos, df_frames, transforms = tfms_test,
                         split='TEST')
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
        print(f"[{epoch} TRAIN] MSE loss = {train_loss:0.5f}")       

        # evaluate
        val_loss, met_ef = evaluate(m, dl_val, device)
        val_loss_ls.append(val_loss)
        print(f"[{epoch} VALID] MSE loss = {val_loss:0.5f}")
        print(f"[{epoch} VALID] EF: ", *[f"{k}={v:0.5f}" for k, v in met_ef.items()])

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
    val_loss, met_ef = evaluate(m, dl_val, device)
    print(f"\tMSE loss = {val_loss:0.5f}")
    print("\tEF: ", *[f"{k}={v:0.5f}" for k, v in met_ef.items()])
    
    print("Test:")
    test_loss, met_ef = evaluate(m, dl_test, device)
    print(f"\tMSE loss = {test_loss:0.5f}")
    print("\tEF: ", *[f"{k}={v:0.5f}" for k, v in met_ef.items()])
       
if __name__=='__main__':
    import argparse
    import sys
    
    parser = argparse.ArgumentParser('Train pda video classifier')
    parser.add_argument('config', type=str, metavar='FILE',
                        help='YAML configuration file.')
    parser.add_argument('--artifact-folder', type=str, metavar='DIR', default=None,
                        help='Directory for model outputs. Overrides "artifact_folder" in config if provided.')
    parser.add_argument('--video-csv', type=str, metavar='FILE', default=None,
                        help='CSV video data file. Overrides "video_csv" in config if provided.')
    parser.add_argument('--frame-csv', type=str, metavar='FILE', default=None,
                        help='CSV frame data file. Overrides "frame_csv" in config if provided.')
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
    if args.video_csv is not None:
        cfg['video_csv'] = args.video_csv
    if args.frame_csv is not None:
        cfg['frame_csv'] = args.frame_csv
    if args.num_heads is not None:
        cfg['vidnet_kwargs']['num_heads'] = args.num_heads
    if args.pooling_method is not None:
        cfg['vidnet_kwargs']['pooling_method'] = args.pooling_method
    if args.device is not None:
        cfg['device'] = args.device
    
    print("Running training script with configuration:")
    print('-'*30)
    yaml.dump(cfg, sys.stdout)
    print('-'*30)

    main(cfg)