import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import timm
from timm import optim, scheduler
import torch
from torch.optim.lr_scheduler import ExponentialLR

from sklearn.model_selection import train_test_split
from sklearn import metrics as skmet
import os
import json

import transforms as my_transforms
from dataset import ImageData

##
## Config

# data paths
frame = '/zfs/wficai/pda/model_data/20221008_frame.csv'
video = '/zfs/wficai/pda/model_data/20221008_video.csv'
study = '/zfs/wficai/pda/model_data/20221008_study.csv'
patient_study = '/zfs/wficai/pda/model_data/20221008_patient_study.csv'
patient = '/zfs/wficai/pda/model_data/20221008_patient.csv'

train = '/zfs/wficai/pda/model_data/train.csv'
val = '/zfs/wficai/pda/model_data/val.csv'
test = '/zfs/wficai/pda/model_data/test.csv'

# load data
df_frame = pd.read_csv(frame)
df_video = pd.read_csv(video)
df_study = pd.read_csv(study)
df_patient_study = pd.read_csv(patient_study)
df_patient = pd.read_csv(patient)

# split data
test_frac = 0.02
val_frac = 0.20 # fraction of training data used for validation
df_patient_train, df_patient_test = train_test_split(df_patient, test_size=test_frac, shuffle=True, stratify=df_patient.patient_type)
df_patient_train, df_patient_val = train_test_split(df_patient_train, test_size=val_frac, shuffle=True, stratify=df_patient_train.patient_type)

# now drop patient_type to avoid incorrect merge with study table
df_patient_train, df_patient_val, df_patient_test = [
    d.drop('patient_type', axis=1) 
    for d in (df_patient_train, df_patient_val, df_patient_test)
]

# merge on frame data
df_train, df_val, df_test = [
    d.merge(df_patient_study, on='patient_id').\
    merge(df_study, on=['patient_type', 'study']).\
    merge(df_video, on=['patient_type', 'study']).\
    merge(df_frame, on=['patient_type', 'external_id'])
    for d in (df_patient_train, df_patient_val, df_patient_test)
]

# ensure that patients are disjoint
train_patient = set(df_train.patient_id)
val_patient = set(df_val.patient_id)
test_patient = set(df_test.patient_id)
assert train_patient.union(val_patient).isdisjoint(test_patient), 'Set of train patients and set of test patients are not disjoint!'

# ensure that studies are disjoint
train_study = set(df_train.study + df_train.patient_type)
val_study = set(df_val.study + df_val.patient_type)
test_study = set(df_test.study + df_test.patient_type)
assert train_study.union(val_study).isdisjoint(test_study), 'Set of train studies and set of test studies are not disjoint!'

# ensure that videos are disjoint
train_vids = set(df_train.external_id + df_train.patient_type)
val_vids = set(df_val.external_id + df_val.patient_type)
test_vids = set(df_test.external_id + df_test.patient_type)
assert train_vids.union(val_vids).isdisjoint(test_vids), 'Set of train videos and set of test videos are not disjoint!'

# ensure that frames are disjoint
train_frames = set(df_train.png_path)
val_frames = set(df_val.png_path)
test_frames = set(df_test.png_path)
assert train_frames.union(val_frames).isdisjoint(test_frames), 'Set of train frames and set of test frames are not disjoint!'

print("All disjoint checks passed")

# save splits
df_train.to_csv(train, index=False)
df_val.to_csv(val, index=False)
df_test.to_csv(test, index=False)