from torchvision import transforms as tfm
import torch

class Transforms:
    def __init__(self, res: int):

        self._tfms_train = tfm.Compose([
            tfm.Resize(res),
            tfm.CenterCrop(res),
            tfm.RandomEqualize(p=0.5),
            tfm.RandAugment(),
            tfm.ToTensor(),
            tfm.RandomErasing(scale=(0.02, 0.1)),
            tfm.Normalize(mean=torch.Tensor([0.4850, 0.4560, 0.4060]), std=torch.Tensor([0.2290, 0.2240, 0.2250])),
            tfm.RandomHorizontalFlip(),
            tfm.RandomRotation((-45, 45)),
            tfm.RandomInvert()
        ])

        self._tfms_test = tfm.Compose([
            tfm.Resize(res),
            tfm.CenterCrop(res),
            tfm.ToTensor(),
            tfm.Normalize(mean=torch.Tensor([0.4850, 0.4560, 0.4060]), std=torch.Tensor([0.2290, 0.2240, 0.2250])),
        ])

    def get_transforms(self, key):
        implemented_keys = ['train', 'test']
        if key == 'train':
            return self._tfms_train
        elif key == 'test':
            return self._tfms_test
        else:
            raise NotImplementedError(f"Transform {key} is not implemented. Choose one of {implemented_keys}.")
