import torch
from torch.utils.data import Dataset, default_collate
from torch.utils.data import DataLoader
from torchvision.io import read_image
import pandas as pd
import random
import copy
from sklearn.model_selection import StratifiedGroupKFold
import numpy as np

class PdaVideos(Dataset):
    """PDA Video dataset used to train video-based model"""
    
    # definte a coding from type, view, and mode names to integers for modeling
    type_map = {'nopda': 0, 'pda': 1}
    view_map = {'nonPDAView': 0, 'pdaRelatedView': 1, 'pdaView': 2}
    mode_map = {'2d': 0, 'color': 1, 'color_compare': 2}
    
    # define inverse maps
    inv_type_map = {v: k for k, v in type_map.items()}
    inv_view_map = {v: k for k, v in view_map.items()}
    inv_mode_map = {v: k for k, v in mode_map.items()}

    def __init__(self, 
                 data: pd.DataFrame, 
                 transforms = None, 
                 split = None,
                 view_filter=None, mode_filter=None, 
                 subset_column=None):
        
        self.data = data
        
        # subset to split
        if split is not None:
            self.data = data.query('Split==@split').copy() 
        
        # optionally subset by subset column
        if subset_column is not None:
            self.data = self.data[self.data[subset_column]].copy()

        # recode targets
        self.data.loc[:,'trg_type'] = self.data['patient_type'].map(self.type_map)
        self.data.loc[:,'trg_view'] = self.data['view'].map(self.view_map)
        self.data.loc[:,'trg_mode'] = self.data['mode'].map(self.mode_map)

        # set transforms
        if transforms is not None:
            self.tfms = transforms

        # apply filters
        if view_filter is not None:
            self.data = self.data.loc[self.data['view'].isin(view_filter)]
        if mode_filter is not None:
            self.data = self.data.loc[self.data['mode'].isin(mode_filter)]
            
        self.data = self.data.reset_index()

        # group to get video-level o data
        self.video_data = self.data[[
            'study', 'patient_id', 'patient_type', 'external_id', 
            'cv_split', 'trg_type', 'trg_view', 'trg_mode'
            ]].drop_duplicates().reset_index()

    def __getitem__(self, index) -> dict:
        row = self.video_data.iloc[index]

        # read and prep the video frames
        is_frame_from_vid = (self.data.patient_type == row['patient_type']) & \
                            (self.data.external_id == row['external_id'])
        frames = self.data.loc[is_frame_from_vid, 'png_path'].tolist()
        frames = sorted(frames)
        
        # read all frames and concatenate into the video
        video = torch.concat([read_image(f)[None] for f in frames])
        
        # apply transforms to entire video
        video = self.tfms(video)

        # build the output dict
        record = dict()
        record['video'] = video
        record['trg_type'] = int(row['trg_type'])
        record['trg_view'] = int(row['trg_view'])
        record['trg_mode'] = int(row['trg_mode'])
        record['study'] = row['study']
        record['patient'] = row['patient_id']
        record['external_id'] = row['external_id']

        return record

    def __len__(self):
        return len(self.video_data)
    
    def cv_dl_gen(self, dataloader_kwargs, tfms):
        num_cv_splits = 10 # TODO: don't hardcode the number of splits!
        for split_ix in range(num_cv_splits):  
            # randomly choose a vlidation index
            val_ix = random.choice([j for j in range(num_cv_splits) if j != split_ix])
            
            # identify which rows are in which split
            test_cond = (self.video_data.cv_split==split_ix)
            val_cond = (self.video_data.cv_split==val_ix)
            train_cond = ~test_cond & ~val_cond
            
            # get the indices of the samples in each split
            train_ix = self.video_data[train_cond].index
            val_ix = self.video_data[val_cond].index
            test_ix = self.video_data[test_cond].index
            
            # get the datasets for each
            d_train = torch.utils.data.Subset(self, train_ix)
            d_val = torch.utils.data.Subset(self, val_ix)
            d_test = torch.utils.data.Subset(self, test_ix)
            
            # set the transforms appropriately
            # first need to make a shallow copy of the training dataset
            # otherwise setting val/test transforms will change training data
            # shallow copy avoids copy the large dataframes (i think)
            d_train.dataset = copy.copy(self)
            
            # get the split-specific transforms and set them
            tfms_train = tfms.get_transforms('train')
            tfms_test = tfms.get_transforms('test')  # same tfms for val/test
            d_train.dataset.tfms = tfms_train
            d_val.dataset.tfms = tfms_test
            d_test.dataset.tfms = tfms_test
            
            # make the dataloaders
            dl_train = DataLoader(d_train, shuffle=True, collate_fn=collate_video, **dataloader_kwargs)
            dl_val = DataLoader(d_val, collate_fn=collate_video, **dataloader_kwargs)
            dl_test = DataLoader(d_test, collate_fn=collate_video, **dataloader_kwargs)
            
            yield split_ix, (dl_train, dl_val, dl_test)

    def crossval_generator(self, k, dataloader_kwargs, tfms):
        """
        Yields dataloaders for train/validation splits.
        Each split contains a non-overlapping set of patients.
        Splits are stratified on video quality.
        """

        cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=42)

        for split_ix, (train_ix, val_ix) in enumerate(
            cv.split(self, self.video_data.patient_type, self.video_data.patient_id)
        ):
            print(split_ix, train_ix.shape, val_ix.shape)

            # get the datasets for each
            d_train = torch.utils.data.Subset(self, train_ix)
            d_val = torch.utils.data.Subset(self, val_ix)
            
            # set the transforms appropriately
            # first need to make a shallow copy of the training dataset
            # otherwise setting val/test transforms will change training data
            # shallow copy avoids copy the large dataframes (i think)
            d_train.dataset = copy.copy(self)
            
            # get the split-specific transforms and set them
            tfms_train = tfms.get_transforms('train')
            tfms_test = tfms.get_transforms('test')  # same tfms for val/test
            d_train.dataset.tfms = tfms_train
            d_val.dataset.tfms = tfms_test
            
            # make the dataloaders
            dl_train = DataLoader(d_train, shuffle=True, collate_fn=collate_video, **dataloader_kwargs)
            dl_val = DataLoader(d_val, collate_fn=collate_video, **dataloader_kwargs)

            yield split_ix, (dl_train, dl_val)

    
class EchoNetVideos(Dataset):
    def __init__(self, video_data: pd.DataFrame, frame_data: pd.DataFrame,
                 transforms, split):
        self.frames = frame_data
        
        # subset to split
        self.data = video_data.query('Split==@split')

        # set transforms
        self.tfms = transforms

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> dict:
        row = self.data.iloc[index]
        frames = self.frames.loc[self.frames.FileName == row['FileName'], 'jpg_path'].tolist()
        frames = sorted(frames)
        
        # read all frames and concatenate into the video
        video = torch.concat([read_image(f)[None] for f in frames])
        
        # apply transforms to entire video
        video = self.tfms(video)

        # create a data record containing the transformed image array and all metadata
        record = dict()
        record['video'] = video
        record['EF'] = row['EF']
        record['ESV'] = row['ESV']/100
        record['EDV'] = row['EDV']/100

        return record
    
def collate_video(batch_list):
    """
    A custom collate function to be passed to the callate_fn argument when creating a pytorch dataloader.
    This is necessary because videos have different lengths. We handle by combining all videos along the time 
    dimension and returning the number of frames in each video.
    """
    vids = torch.concat([b['video'] for b in batch_list])
    num_frames = [b['video'].shape[0] for b in batch_list]

    record = {
        'video': vids,
        'num_frames': num_frames
    }

    # use pytorch's default collate function for remaining items
    for b in batch_list:
        b.pop('video')
    record.update(default_collate(batch_list))

    return record


