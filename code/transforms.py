from torchvision import transforms as tfm
import torch
from torch import nn
import torch.nn.functional as F

resize_factor = 1.1

class VideoTransforms:
    def __init__(self, res: int, 
                 mean=torch.Tensor([0.4850, 0.4560, 0.4060]), 
                 std=torch.Tensor([0.2290, 0.2240, 0.2250]),
                 time_downsample_kwargs = None):
        
        if time_downsample_kwargs is None:
            time_downsample_kwargs = {'method': 'random', 'num_frames': 16}
        
        # training transforms
        self._tfms_train = tfm.Compose([
            DownsampleTime(**time_downsample_kwargs),
            tfm.RandAugment(),
            tfm.ConvertImageDtype(torch.float32),
            tfm.Resize((res, res)),
            # tfm.RandomErasing(scale=(0.02, 0.1)),
            tfm.Normalize(mean=mean, std=std),
            # tfm.RandomRotation((-10, 10))
        ])

        # eval/testing transforms
        self._tfms_test = tfm.Compose([
            tfm.ConvertImageDtype(torch.float32),
            tfm.Resize((res, res)),
            tfm.Normalize(mean=mean, std=std),
        ])
        
        # plotting transforms
        self._tfms_plot = tfm.Compose([
            tfm.ConvertImageDtype(torch.float32),
            tfm.Resize((res, res)),
        ])

    def get_transforms(self, key):
        implemented_keys = ['train', 'test', 'plot']
        if key == 'train':
            return self._tfms_train
        elif key == 'test':
            return self._tfms_test
        elif key == 'plot': 
            return self._tfms_plot
        else:
            raise NotImplementedError(f"Transform {key} is not implemented. Choose one of {implemented_keys}.")


class CustomCrop(nn.Module):
    def __init__(self, x0, y0, w, h):
        super().__init__()

        self.x_slice = slice(x0, x0 + w)
        self.y_slice = slice(y0, y0 + h)

    def forward(self, img):
        return img[..., self.y_slice, self.x_slice]
    
class DownsampleTime(nn.Module):
    def __init__(self, method, num_frames, stride=1):
        super().__init__()
        assert method in ['random', 'contiguous'], "Method must be 'random' or 'contiguous'"
        self.method = method
        self.num_frames = num_frames
        self.stride = stride
        
        if method=='contiguous':
            print("Warning: the contiguous method is untested.")
            self.clip_span = num_frames + stride * (num_frames - 1)
        
    def forward(self, vid):        
        if self.method=='random':
            tix = torch.randint(low=0, high=vid.shape[0], size=(self.num_frames,))
        elif self.method=='contiguous':
            assert self.clip_span < vid.shape[0], \
                f"The requested num_frames {self.num_frames} and stride {self.stride}" + \
                f"imply a clip span {self.clip_span} that is larger than" + \
                f"the length of the input video {vid.shape[0]}"
            start_ix = torch.randint(low=0, high=vid.shape[0] - self.clip_span, size=(1,))
            tix = slice(start_ix, start_ix+self.clip_span, self.stride)
        
        # vid: [L, C, H, W]
        return vid[tix]