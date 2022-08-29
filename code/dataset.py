import torch
from torch.utils.data import Dataset, default_collate
from torchvision.io import read_image
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


class PDAData(Dataset, ABC):
    """Image CSV Dataset"""

    type_map = {'nopda': 0, 'pda': 1}
    view_map = {'nonPDAView': 0, 'pdaRelatedView': 1, 'pdaView': 2}
    mode_map = {'2d': 0, 'color': 1, 'color_compare': 2}

    inv_type_map = {v: k for k, v in type_map.items()}
    inv_view_map = {v: k for k, v in view_map.items()}
    inv_mode_map = {v: k for k, v in mode_map.items()}

    def __init__(self, data: pd.DataFrame, transforms, view_filter=None, mode_filter=None):
        self.data = data

        # recode targets
        self.data.loc[:, 'trg_type'] = self.data.loc[:, 'patient_type'].map(PDAData.type_map)
        self.data.loc[:, 'trg_view'] = self.data.loc[:, 'view'].map(PDAData.view_map)
        self.data.loc[:, 'trg_mode'] = self.data.loc[:, 'mode'].map(PDAData.mode_map)

        # set transforms
        self.tfms = transforms

        # apply filters
        if view_filter is not None:
            self.data = self.data.loc[self.data['view'].isin(view_filter)]
        if mode_filter is not None:
            self.data = self.data.loc[self.data['mode'].isin(mode_filter)]

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, index) -> dict:
        pass


class ImageData(PDAData):
    """Image CSV Dataset"""

    def __init__(self, data: pd.DataFrame, transforms, view_filter=None, mode_filter=None):
        super().__init__(data, transforms, view_filter, mode_filter)

    def __getitem__(self, index) -> dict:
        row = self.data.iloc[index]

        record = dict()
        record['img'] = self.tfms(Image.open(row['png_path']))
        record['trg_type'] = int(row['trg_type'])
        record['trg_view'] = int(row['trg_view'])
        record['trg_mode'] = int(row['trg_mode'])
        record['video'] = row['external_id']
        record['study'] = row['study']
        record['patient'] = row['patient_id']

        return record

    def __len__(self):
        return len(self.data)

    @staticmethod
    def display_batch(batch, n_cols=8, height=20):
        num_images = batch['img'].shape[0]
        n_rows = num_images // n_cols
        if n_rows * n_cols < num_images:
            n_rows += 1

        fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, sharex=True, sharey=True)
        fig.set_size_inches(height, height * n_rows / n_cols)

        axs = axs.flatten()

        for i, img in enumerate(batch['img']):
            if img.shape[0] == 3:
                img = img.permute(1, 2, 0)

            axs[i].imshow(img.squeeze(), cmap='Greys_r')
            axs[i].axis('off')

            patient = batch['patient'][i]
            type = ImageData.inv_type_map[batch['trg_type'][i].item()]
            view = ImageData.inv_view_map[batch['trg_view'][i].item()]
            mode = ImageData.inv_mode_map[batch['trg_mode'][i].item()]

            title = f"{patient}\n{type} - {view} - {mode}"
            axs[i].set_title(title)

        plt.tight_layout()
        plt.show()


class VideoData(PDAData):
    """Image CSV Dataset"""

    def __init__(self, data: pd.DataFrame, transforms, view_filter=None, mode_filter=None):
        super().__init__(data, transforms, view_filter, mode_filter)

        self.video_data = self.data[['study', 'patient_id', 'patient_type', 'external_id', 'mode', 'trg_type', 'trg_view', 'trg_mode']].drop_duplicates()

    def __getitem__(self, index) -> dict:
        row = self.video_data.iloc[index]

        # read and prep the video frames
        is_frame_from_vid = (self.data.patient_type == row['patient_type']) & \
                            (self.data.external_id == row['external_id'])
        frames = self.data.loc[is_frame_from_vid, 'png_path'].tolist()
        frames = sorted(frames)
        video = torch.concat([read_image(f)[None] for f in frames])

        # build the output dict
        record = dict()
        record['video'] = self.tfms(video)
        record['trg_type'] = int(row['trg_type'])
        record['trg_view'] = int(row['trg_view'])
        record['trg_mode'] = int(row['trg_mode'])
        record['study'] = row['study']
        record['patient'] = row['patient_id']

        return record

    def __len__(self):
        return len(self.video_data)

    @staticmethod
    def collate(batch_list):
        # handle collation of videos with different lengths
        # by concat along the time dimension and recording where videos start and end
        vids = torch.concat([b['video'] for b in batch_list])
        num_frames = [b['video'].shape[0] for b in batch_list]
        start_ix = 0

        # create a mask tensor for each video in the batch
        # use slices for conveniently pulling out frames from a particular video
        slices = []
        for n in num_frames:
            end_ix = start_ix + n
            slices.append(slice(start_ix, end_ix))
            start_ix = end_ix

        mask = torch.zeros((vids.shape[0], len(batch_list)), dtype=torch.bool)
        for ix in range(len(batch_list)):
            mask[slices[ix], ix] = 1

        record = {
            'video': vids,
            'mask': mask
        }

        # use default collate function for remaining items
        for b in batch_list:
            b.pop('video')
        record.update(default_collate(batch_list))

        return record
