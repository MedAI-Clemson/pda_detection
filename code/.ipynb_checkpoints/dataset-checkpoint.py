from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from torch import nn


class ImageData(Dataset):
    """Image CSV Dataset"""

    type_map = {'nopda': 0, 'pda': 1}
    view_map = {'nonPDAView': 0, 'pdaRelatedView': 1, 'pdaView': 2}
    mode_map = {'2d': 0, 'color': 1, 'color_compare': 2}
    
    inv_type_map = {v:k for k,v in type_map.items()}
    inv_view_map = {v:k for k,v in view_map.items()}
    inv_mode_map = {v:k for k,v in mode_map.items()}

    def __init__(self, data: pd.DataFrame, transforms, view_filter=None, mode_filter=None):
        self.data = data

        # recode targets
        self.data['trg_type'] = self.data['patient_type'].map(ImageData.type_map)
        self.data['trg_view'] = self.data['view'].map(ImageData.view_map)
        self.data['trg_mode'] = self.data['mode'].map(ImageData.mode_map)

        # set transforms
        self.tfms = transforms

        # apply filters
        if view_filter is not None:
            self.data = self.data.loc[self.data['view'].isin(view_filter)]
        if mode_filter is not None:
            self.data = self.data.loc[self.data['mode'].isin(mode_filter)]

    def __len__(self) -> int:
        return len(self.data)

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
                img = img.permute(1,2,0)

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


class CustomCrop(nn.Module):
    def __init__(self, x0, y0, w, h):
        super().__init__()

        self.x_slice = slice(x0, x0 + w)
        self.y_slice = slice(y0, y0 + h)

    def forward(self, img):
        return img[..., self.y_slice, self.x_slice]