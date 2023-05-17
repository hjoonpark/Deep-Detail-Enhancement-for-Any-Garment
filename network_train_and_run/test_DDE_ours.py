from __future__ import division
from __future__ import print_function

import time
import argparse
import os
import glob
import torch
import shutil
from torchvision import transforms

from PIL import Image
from random import shuffle
import torch.optim as optim

from models import Generator_CNNCIN_nvidia as Generator_CNNCIN
from VGG_Model import vgg_mean, vgg_std
from Losses import BatchFeatureLoss_Model
from torchvision.utils import save_image

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from dataset import DatasetNormalmaps
from ddp import DDPWraper

import numpy as np
def save_image_plot(I_ins, I_outs, I_gts, save_path):
    fname = os.path.basename(save_path)
    b = 0
    std = np.array(vgg_std)
    mean = np.array(vgg_mean)

    fig = plt.figure(figsize=(15, 11))
    ax = fig.add_subplot(231)
    ax.set_xlabel("Input low-res")
    I_in = I_ins[b].cpu().permute(1, 2, 0).numpy()
    I_in = I_in*std + mean
    ax.imshow(I_in)
    ax.set_title("{} ({:.2f}, {:.2f}) [{:.2f}, {:.2f}]".format(I_in.shape, I_in.min(), I_in.max(), I_in.mean(), I_in.std()))

    ax = fig.add_subplot(232)
    ax.set_xlabel("Reconstruction high-res")
    I_out = I_outs[b].cpu().permute(1, 2, 0).detach().numpy()
    I_out = I_out*std + mean
    # I_out = (I_out-I_out.min())/(I_out.max()-I_out.min())
    ax.imshow(I_out)
    ax.set_title("{} ({:.2f}, {:.2f}) [{:.2f}, {:.2f}]".format(I_out.shape, I_out.min(), I_out.max(), I_out.mean(), I_out.std()))
    
    ax = fig.add_subplot(233)
    ax.set_xlabel("Target high-res")
    I_gt = I_gts[b].cpu().permute(1, 2, 0).detach().numpy()
    I_gt = I_gt*std + mean
    # I_gt = (I_gt-I_gt.min())/(I_gt.max()-I_gt.min())
    ax.imshow(I_gt)
    ax.set_title("{} ({:.2f}, {:.2f}) [{:.2f}, {:.2f}]".format(I_gt.shape, I_gt.min(), I_gt.max(), I_gt.mean(), I_gt.std()))

    L_diff = I_out-I_in
    H_diff = I_out-I_gt
    HL_diff = I_gt-I_in
    diff_max = max(L_diff.max(), H_diff.max())
    diff_min = min(L_diff.min(), H_diff.min())

    ax = fig.add_subplot(234)
    ax.set_xlabel("Recon diff: low-res")
    L_diff = (L_diff-diff_min)/(diff_max-diff_min)
    ax.imshow(L_diff)
    ax.set_title("Error: mean={:.4f}, std={:.4f} | ({:.2f}, {:.2f})".format(L_diff.mean(), L_diff.std(), L_diff.min(), L_diff.max()))
    
    ax = fig.add_subplot(235)
    ax.set_xlabel("Diff: high- and low-res")
    HL_diff = (HL_diff-diff_min)/(diff_max-diff_min)
    ax.imshow(HL_diff)
    ax.set_title("Error: mean={:.4f}, std={:.4f} | ({:.2f}, {:.2f})".format(HL_diff.mean(), HL_diff.std(), HL_diff.min(), HL_diff.max()))
    
    ax = fig.add_subplot(236)
    ax.set_xlabel("Recon diff: high-res")
    H_diff = (H_diff-diff_min)/(diff_max-diff_min)
    ax.imshow(H_diff)
    ax.set_title("Error: mean={:.4f}, std={:.4f} | ({:.2f}, {:.2f})".format(H_diff.mean(), H_diff.std(), H_diff.min(), H_diff.max()))

    plt.suptitle(fname)
    plt.tight_layout()
    # save_path = os.path.join(plot_dir, "test_image.png")
    plt.savefig(save_path, dpi=150)
    print("image plot saved:", save_path)
    
# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--patience', type=int, default=100000, help='Patience')
parser.add_argument('--model-path', type=str, default="")

args = parser.parse_args()
args.model_path = "output_ours/ckp/c_best_model_1100.ckp"

batch_size = 1
dataset_te = DatasetNormalmaps(is_train=False)
data_loader_te = torch.utils.data.DataLoader(dataset_te, batch_size=batch_size)
device = torch.device("cuda:0")

def load_model(model, optimizer, model_path):
    try:
        model = model.module
    except:
        pass

    try:
        loaded = torch.load(model_path)
        model_state = loaded["model"]
        model.load_state_dict(model_state, strict=False)

        # optimizer_state = loaded["optimizer"]
        # optimizer.load_state_dict(optimizer_state)

        test_loss = loaded["loss"]
        itt = loaded["itter"]

        print(":: loading model: {}".format(model_path))
        print("  loaded succesfully!")
        print("  - itt: {}, test_loss: {}".format(itt, test_loss))

        return itt, test_loss
    except Exception as e:
        print(":: failed to load {} - architecture mismatch! Initializing new instead".format(model_path))
        print(e)

Patch_H = 512
Patch_W = 512

def saveBImg(itt, DirList, batchIDs, imgBatch, maskBatch, saveDir):
    c = 0
    for id in range(len(batchIDs)):
        imgN = DirList[id]
        bname = os.path.basename(imgN).split(".png")[0]
        saveN = os.path.join(saveDir, f"{itt:08d}_{bname}.png")
        print("saveN:", saveN)
        rst = imgBatch[c, 0:3, :, :] * torch.tensor(vgg_std).view(-1, 1, 1).to(device) + \
              torch.tensor(vgg_mean).view(-1, 1, 1).to(device)
        # rst = imgBatch[c, 0:3, :, :]
        # rst = (rst-rst.min())/(rst.max()-rst.min())
        save_image(rst, fp=saveN)

        if imgBatch.shape[0] < c-1:
            c = c+1
        else:
            break
    return

"""
Load train dataset
"""
in_dir = "nvidia_data/BakedUVMaps"
"""
Load test dataset
"""
#model = Generator_CNN_Cat(3+len(test_InfoDir)*3, 3).to(device)
NumMat = 1
model = Generator_CNNCIN(inDim=3, outDim=3, styleNum=NumMat).to(device)

world_size = 1
if world_size > 1:
    ddp = DDPWraper(rank=0, world_size=2)
    model = ddp.setup_model(model)
    model_module = model.module
else:
    model = model.to(device)
    model_module = model
    
lossFunc = BatchFeatureLoss_Model(device=device, c_alpha=1., s_beta=1.e4, s_layWei=[1., 1., 1., 1., 1.]).to(device)

plot_dir = "output_ours/test/plot"
save_dir = "output_ours/test/predictions"
os.makedirs(plot_dir, exist_ok=1)
os.makedirs(save_dir, exist_ok=1)
losses = {"frame": [], "c_Loss": [], "s_Loss": []}

def model_test(data):
    model.eval()
    with torch.no_grad():
        lres = data["lres"].to(device)
        hres = data["hres"].to(device)
        mask = data["mask"].to(device)
        frame = data["frame"][0]
        Out = model(lres)
        c_Loss, s_Loss = lossFunc(X=Out, SG=hres, CX=hres[:, 0:3, :, :], MX=mask)
        Loss = c_Loss + s_Loss
        Loss = Loss

        losses["c_Loss"].append(c_Loss.item())
        losses["s_Loss"].append(s_Loss.item())
        losses["frame"].append(frame)

        # save normal map
        out_fname = f"{frame}".replace(".npy", ".png")
        saveN = os.path.join(save_dir, out_fname)
        rst = Out[0, 0:3, :, :] * torch.tensor(vgg_std).view(-1, 1, 1).to(device) + \
              torch.tensor(vgg_mean).view(-1, 1, 1).to(device)
        # rst = (rst-rst.min())/(rst.max()-rst.min())
        print("saveN:", saveN, rst.shape, rst.min().item(), rst.max().item())
        save_image(rst, fp=saveN)

        # save plot
        save_path = os.path.join(plot_dir, out_fname)
        save_image_plot(lres, Out, hres, save_path)

        return Loss

if args.model_path != "":
    model_path = args.model_path
    try:
        ckp = torch.load(args.model_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(ckp['model'])
        print(":: loading model: {}".format(model_path))
        print("  loaded succesfully!")
    except Exception as e:
        print(":: failed to load {} - architecture mismatch! Initializing new instead".format(model_path))
        print(e)
    
IFSumWriter = False

for data_idx, data in enumerate(data_loader_te):
    test_loss = model_test(data)

print("### DONE")