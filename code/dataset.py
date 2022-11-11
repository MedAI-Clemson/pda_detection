import torch
from torch.utils.data import Dataset, default_collate
from torchvision.io import read_image
import pandas as pd

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

    def __init__(self, data: pd.DataFrame, transforms, split, view_filter=None, mode_filter=None, subset_column=None):
        
        # subset to split
        self.data = data.query('Split==@split').copy() 
        
        # optionally subset by subset column
        if subset_column is not None:
            self.data = self.data[self.data[subset_column]].copy()

        # recode targets
        self.data.loc[:,'trg_type'] = self.data['patient_type'].map(self.type_map)
        self.data.loc[:,'trg_view'] = self.data['view'].map(self.view_map)
        self.data.loc[:,'trg_mode'] = self.data['mode'].map(self.mode_map)

        # set transforms
        self.tfms = transforms

        # apply filters
        if view_filter is not None:
            self.data = self.data.loc[self.data['view'].isin(view_filter)]
        if mode_filter is not None:
            self.data = self.data.loc[self.data['mode'].isin(mode_filter)]

        # group to get video-level o data
        self.video_data = self.data[['study', 'patient_id', 'patient_type', 'external_id', 
                                     'trg_type', 'trg_view', 'trg_mode']].drop_duplicates()

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