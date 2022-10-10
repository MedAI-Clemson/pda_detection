import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import timm
from timm import optim, scheduler
import torch
from torch.optim.lr_scheduler import ExponentialLR
from sklearn import metrics as skmet
import os
import json

# locally defined
import transforms as my_transforms
from dataset import VideoData
import models


def train_one_epoch(model, optimizer, train_dataloader, loss_function, device):
    model.train()

    num_steps_per_epoch = len(train_dataloader)

    losses = []
    for ix, batch in enumerate(train_dataloader):
        torch.autograd.set_detect_anomaly(True)
        inputs = batch['video'].to(device)
        num_frames = batch['num_frames']
        targets = batch['trg_type'].to(device).type(torch.float32)
        outputs, _ = model(inputs, num_frames)
        loss = loss_function(outputs.squeeze(), targets.squeeze())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss.detach().item())
        print(f"\tBatch {ix+1} of {num_steps_per_epoch}. Loss={loss.detach().item():0.3f}", end='\r')
    
    print(' '*100, end='\r')
        
    return np.mean(losses)  
            
def evaluate(model, test_dataloader, loss_function, device):
    model.eval()

    num_steps_per_epoch = len(test_dataloader)

    patient_ls = []
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
            loss = loss_function(outputs.squeeze(), targets.squeeze())
            
        losses.append(loss.detach().item())
        
    metrics = compute_metrics(np.concatenate(target_ls), np.concatenate(output_ls))
    return np.mean(losses), metrics

def compute_metrics(y_true, y_pred):
    mets = dict()
    
    y_pred = 1/(1+np.exp(-y_pred))
    y_pred_cls = (y_pred>0.5).astype(int)
    
    mets['roc_auc'] = skmet.roc_auc_score(y_true, y_pred)
    mets['average_precision'] = skmet.average_precision_score(y_true, y_pred)
    mets['accuracy'] = skmet.accuracy_score(y_true, y_pred_cls)
    mets['sensitivity'] = skmet.recall_score(y_true, y_pred_cls)
    mets['specificity'] = skmet.recall_score(y_true, y_pred_cls, pos_label=0)
    
    return mets

def get_video_classifier(attn, prob):
    if attn=='none':
        if prob=='PI':
            return models.VideoClassifier
        else:
            raise NotImplementedError("The none_LSTM model has not been implemented")
    elif attn=='PI':
        if prob=='PI':
            return models.VideoClassifier_PIattn
        else:
            raise NotImplementedError("The PI_LSTM model has not been implemented")
    elif attn=="LSTM":
        if prob=='PI':
            return models.VideoClassifier_LSTMattn
        else:
            raise NotImplementedError("The LSTM_LSTM model has not been implemented")

def main(cfg):
    device = torch.device(cfg['device'])

    # transforms
    tfms = my_transforms.VideoTransforms(cfg['config_pretrain']['res'], cfg['time_downsample_factor'], cfg['time_downsample_method'])
    tfms_train = tfms.get_transforms(cfg['video_transforms']['train'])
    tfms_test = tfms.get_transforms(cfg['video_transforms']['test'])

    # load splits
    df_train = pd.read_csv(cfg['in_paths']['train'])
    df_val = pd.read_csv(cfg['in_paths']['val'])

    # create datasets
    d_train = VideoData(df_train, transforms = tfms_train, 
                        mode_filter = cfg['mode_filter'], view_filter = cfg['view_filter'])
    dl_train = DataLoader(d_train, batch_size=cfg['bs_train'], num_workers=cfg['num_workers'], 
                          shuffle=True, collate_fn=VideoData.collate)
    d_val = VideoData(df_val, transforms = tfms_test, 
                    mode_filter = cfg['mode_filter'], view_filter = cfg['view_filter'])
    dl_val= DataLoader(d_val, batch_size=cfg['bs_val'], num_workers=cfg['num_workers'],
                       collate_fn=VideoData.collate)

    print("Train data size:", len(d_train))
    print("Validation data size:", len(d_val))


    # classifier network
    m_frames = timm.create_model(cfg['config_pretrain']['model'], pretrained=cfg['pretrained'],
                                 checkpoint_path = f"{cfg['pretrain_folder']}/model_checkpoint.ckpt",
                                 num_classes=1, in_chans=3, drop_rate=cfg['dropout'])
    m_frames.to(device)
    
    # create video model
    video_classifier = get_video_classifier(cfg['attn'], cfg['prob'])
    m = video_classifier(m_frames, encoder_frozen=True, frame_classifier_frozen=False)

    # fit
    optimizer = optim.AdamP(m.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = ExponentialLR(optimizer, gamma=cfg['lr_gamma'])
    loss_function = torch.functional.F.binary_cross_entropy
    
    # pre-training validation
    print(evaluate(m, dl_val, loss_function, device))

    train_loss_ls = []
    test_loss_ls = []
    metrics_ls = []

    best_test_loss = 1000
    is_frozen = True
    for epoch in range(cfg['num_epochs']):
        print("-"*40)
        print(f"Epoch {epoch+1} of {cfg['num_epochs']}:")
        
        # maybe unfreeze 
        if epoch >= cfg['unfreeze_after_n'] and is_frozen:
            print("Unfreezing model encoder.")
            is_frozen=False
            for p in m.encoder.parameters():
                p.requires_grad = True
                
            # set all learning rates to the lower lr_unfrozen learning rate
            for g in optimizer.param_groups:
                g['lr'] = cfg['lr_unfrozen']

        # train for a single epoch
        train_loss = train_one_epoch(m, optimizer, dl_train, loss_function, device)
        train_loss_ls.append(train_loss)
        print(f"Training:")
        print(f"\tcross_entropy = {train_loss:0.3f}")       

        # evaluate
        test_loss, metrics = evaluate(m, dl_val, loss_function, device)
        test_loss_ls.append(test_loss)
        metrics_ls.append(metrics)
        print(f"Test:")
        print(f"\tcross_entropy = {test_loss:0.3f}")
        print(f"\tmetrics:")
        for k, v in metrics.items():
            print(f"\t\t{k} = {v:0.3f}")

        if test_loss < best_test_loss:
            torch.save(m.state_dict(), f"{cfg['artifact_folder']}/model_checkpoint_video.ckpt")
            best_test_loss = test_loss
            
        scheduler.step()
        
if __name__=='__main__':
    import argparse
    import config
    from pathlib import Path
    from config import config_video as cfg
    
    parser = argparse.ArgumentParser(description='Train a video classifier')
    parser.add_argument('--attn', type=str, choices = ('none', 'PI', 'LSTM'), default='none',
                        help='Method used to compute frame attention.')
    parser.add_argument('--prob', type=str, choices = ('PI', 'LSTM'), default='PI',
                        help='Method used to compute frame probabilities.')
    parser.add_argument('--pretrain-folder', type=str, metavar='DIR', required=True,
                        help='path to pretrained frame classifier model artifact folder')
    parser.add_argument('--artifact-folder', type=str, metavar='DIR', required=True,
                        help='path to artifact folder')

    args = parser.parse_args()
    
    cfg['attn'] = args.attn
    cfg['prob'] = args.prob
    cfg['time_downsample_method'] = 'contiguous' if (args.attn=='LSTM' or args.prob=='LSTM') else 'random'
    print(f"Time downsampling method: {cfg['time_downsample_method']}")
    cfg['artifact_folder'] = args.artifact_folder
    cfg['pretrain_folder'] = args.pretrain_folder
    with open(args.pretrain_folder + '/config.json', 'r') as f:
        cfg['config_pretrain'] = json.load(f)
    
    # save the config file to the artifact folder
    os.makedirs(cfg['artifact_folder'], exist_ok=True)
    with open(cfg['artifact_folder'] + 'config.json', 'w') as f: 
        json.dump(cfg, f, indent=2)

    main(cfg)