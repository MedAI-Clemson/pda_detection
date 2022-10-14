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

import transforms as my_transforms
from dataset import EchoNetFrames


def train_one_epoch(model, optimizer, train_dataloader, loss_function, device):
    model.train()

    num_steps_per_epoch = len(train_dataloader)

    losses = []
    for ix, batch in enumerate(train_dataloader):
        inputs = batch['img'].to(device)
        targets = batch['trg_type'].to(device).type(torch.float32)
        outputs = model(inputs)
        loss = loss_function(outputs, targets[...,None])

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss.detach().item())
        print(f"\tBatch {ix+1} of {num_steps_per_epoch}. Loss={loss.detach().item():0.3f}", end='\r')
    
    print(' '*100, end='\r')
        
    return np.mean(losses)    
            
def evaluate(model, val_dataloader, loss_function, device):
    model.eval()

    num_steps_per_epoch = len(val_dataloader)

    patient_ls = []
    target_ls = []
    output_ls = []
    losses = []
    for ix, batch in enumerate(val_dataloader):
        inputs = batch['img'].to(device)
        targets = batch['trg_type'].to(device).type(torch.float32)
        target_ls.append(targets.cpu().numpy())
        
        with torch.no_grad():
            outputs = model(inputs)
            output_ls.append(outputs.cpu().numpy())
            loss = loss_function(outputs, targets[...,None])
            
        losses.append(loss.detach().item())
        
    metrics = compute_metrics(np.concatenate(target_ls), np.concatenate(output_ls))
    return np.mean(losses), metrics

def compute_metrics(y_true, y_pred):
    mets = dict()
    
    y_pred = 1/(1+np.exp(-y_pred))
    y_pred_cls = (y_pred>0.5).astype(int)
    
    mets['r2'] = skmet.r2_score(y_true, y_pred)
    mets['average_precision'] = skmet.average_precision_score(y_true, y_pred)
    mets['accuracy'] = skmet.accuracy_score(y_true, y_pred_cls)
    mets['sensitivity'] = skmet.recall_score(y_true, y_pred_cls)
    mets['specificitay'] = skmet.recall_score(y_true, y_pred_cls, pos_label=0)
    
    return mets

def main(cfg):
    os.makedirs(cfg['artifact_folder'], exist_ok=True)

    # save the config file to the artifact folder
    with open(cfg['artifact_folder'] + '/config.json', 'w') as f: 
        json.dump(cfg, f, indent=4)

    device = torch.device(cfg['device'])

    # transforms
    tfms = my_transforms.ImageTransforms(cfg['res'])
    tfms_train = tfms.get_transforms(cfg['transforms']['train'])
    tfms_test = tfms.get_transforms(cfg['transforms']['test'])

    # load data
    df_train = pd.read_csv(cfg['in_paths']['train'])
    df_val = pd.read_csv(cfg['in_paths']['val'])
    df_frames = pd.read_csv(cfg['in_paths']['frames'])
    
    # create datasets
    d_train = EchoNetFrames(df_train, transforms = tfms_train, 
                        mode_filter = cfg['mode_filter'], view_filter = cfg['view_filter'])
    dl_train = DataLoader(d_train, batch_size=cfg['bs_train'], num_workers=cfg['num_workers'], shuffle=True)

    d_val = ImageData(df_val, transforms = tfms_test, 
                      mode_filter = cfg['mode_filter'], view_filter = cfg['view_filter'])
    dl_val= DataLoader(d_val, batch_size=cfg['bs_val'], num_workers=cfg['num_workers'])

    print("Train data size:", len(d_train))
    print("Validation data size:", len(d_val))

    # classifier network
    m = timm.create_model(cfg['model'], pretrained=cfg['pretrained'], num_classes=1, in_chans=3, drop_rate=cfg['dropout'])
    m.to(device)

    # freeze model weights
    # don't freeze classifier or first conv/bn
    for layer in list(m.children())[2:-1]:
        for p in layer.parameters():
            p.requires_grad = False
    is_frozen=True

    # fit
    optimizer = optim.AdamP(m.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = ExponentialLR(optimizer, gamma=cfg['lr_gamma'])
    loss_function = torch.functional.F.binary_cross_entropy_with_logits

    train_loss_ls = []
    test_loss_ls = []
    metrics_ls = []

    best_test_loss = 1000
    for epoch in range(cfg['num_epochs']):
        print("-"*40)
        print(f"Epoch {epoch+1} of {cfg['num_epochs']}:")

        # maybe unfreeze 
        if epoch >= cfg['unfreeze_after_n'] and is_frozen:
            print("Unfreezing model encoder.")
            is_frozen=False
            for p in m.parameters():
                p.requires_grad = True

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
            torch.save(m.state_dict(), f"{cfg['artifact_folder']}/model_checkpoint.ckpt")
            best_test_loss = test_loss

        scheduler.step()
        
if __name__=='__main__':
    from config import config_pretrain as cfg
    import argparse
    parser = argparse.ArgumentParser('Pretrain frame classifier')
    parser.add_argument('--artifact-folder', type=str, metavar='DIR', required=True,
                        help='path to artifact folder')
    args = parser.parse_args()
    
    cfg['artifact_folder'] = args.artifact_folder
    main(cfg)