import random
import numpy as np
import os
import torch
import torchvision
import glob
import json
from PIL import Image
from torchvision import transforms
from VGG_Model import vgg_mean, vgg_std

def stat(x):
    out_str = "{}, min/max=({:.2f}, {:.2f}), mean={:.2f}, std={:.2f}".format(x.shape, x.min(), x.max(), x.mean(), x.std())
    return out_str

class DatasetNormalmaps(torch.utils.data.Dataset):
    def __init__(self, is_train):
        in_dir = "nvidia_data/normalmaps_0_1"
        self.frames = []
        self.hres = []
        self.lres = []
        self.masks = []

        img_normalize = transforms.Compose([transforms.Normalize(vgg_mean, vgg_std)])
        to_tensor = transforms.Compose([transforms.ToTensor()])

        if is_train:
            folders = ["amazement", "pain"]
        else:
            folders = ["anger", "fear"]
        
        for folder in folders:
            hpaths = sorted(glob.glob(os.path.join(in_dir, folder, "hires*")))
            for hpath in hpaths:
                bname = os.path.basename(hpath)
                lpath = os.path.join(in_dir, folder, bname.replace("hires", "lores"))

                lx = img_normalize(torch.from_numpy(np.load(lpath)))
                hx = img_normalize(torch.from_numpy(np.load(hpath)))

                self.lres.append(lx)
                self.hres.append(hx)
                self.masks.append(torch.ones((1, lx.shape[1], lx.shape[2])))
                self.frames.append('{}_{}'.format(folder, bname))

        print("Dataset {}: {} frames".format("train" if is_train else "test", len(self.frames)))
        lres_ = torch.cat(self.lres).squeeze().numpy().reshape((3, -1))
        hres_ = torch.cat(self.hres).squeeze().numpy().reshape((3, -1))
        print("lres: ")
        print("min={}, max={}".format(lres_.min(1), lres_.max(1)))
        print("mean={}, std={}".format(lres_.mean(1), lres_.std(1)))
        print("hres: ")
        print("min={}, max={}".format(hres_.min(1), hres_.max(1)))
        print("mean={}, std={}".format(hres_.mean(1), hres_.std(1)))

    def __getitem__(self, index):
        frame = self.frames[index]
        lres = self.lres[index]
        hres = self.hres[index]
        mask = self.masks[index]

        out = {"frame": frame, "lres": lres, "hres": hres, "mask": mask}
        return out

    def __len__(self):
        return len(self.frames)