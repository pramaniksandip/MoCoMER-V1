import torchvision.transforms as tr
from torch.utils.data.dataset import Dataset
import random
from PIL import ImageFilter, ImageOps, Image

from .transforms import ScaleAugmentation, ScaleToLimitRange

K_MIN = 0.7
K_MAX = 1.4

H_LO = 16
H_HI = 256
W_LO = 16
W_HI = 1024

class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img
        
class CROHMEDataset(Dataset):
    def __init__(self, ds, is_train: bool, scale_aug: bool) -> None:
        super().__init__()
        self.ds = ds

        trans_list = []
        if is_train and scale_aug:
            trans_list.append(ScaleAugmentation(K_MIN, K_MAX))

        pil_transformation = []

        pil_transformation += [
            GaussianBlur(p=1.0),
            Solarization(p=0.0),
            tr.ToTensor(),

        ]
        self.blur_transform = tr.Compose(pil_transformation)
        trans_list += [
            ScaleToLimitRange(w_lo=W_LO, w_hi=W_HI, h_lo=H_LO, h_hi=H_HI),
           
        ]
        self.transform = tr.Compose(trans_list)

    def __getitem__(self, idx):
        fname, img, caption = self.ds[idx]

        img = [self.transform(im) for im in img]
        img = [Image.fromarray(im) for im in img]
        img = [self.blur_transform(im) for im in img]
        
       # print(img[0].max(), img[0].min())
        #print(type(img[0]))
        #print(img[0].shape)

        return fname, img, caption

    def __len__(self):
        return len(self.ds)
