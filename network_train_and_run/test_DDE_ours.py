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

from models import Generator_CNNCIN
from VGG_Model import vgg_mean, vgg_std
from Losses import BatchFeatureLoss_Model
from torchvision.utils import save_image

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

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
args.model_path = "output_ours/ckp/c_best_model_29000.ckp"

#USE_CUDA = False
USE_CUDA = torch.cuda.is_available()
print("balin-->", USE_CUDA)
device = torch.device("cuda:0" if USE_CUDA else "cpu")

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

NumMat = 1

Img_transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(vgg_mean, vgg_std)])
# Mask_transform = transforms.Compose([transforms.Resize((Patch_H, Patch_W)), transforms.ToTensor()])
Mask_transform = transforms.Compose([transforms.ToTensor()])

def get_fileSet_list(dir, withF=[]):
    if len(withF) == 0:
        return glob.glob(os.path.join(dir, "*"))
    else:
        fL = []
        for wf in withF:
            fL = fL + glob.glob(os.path.join(dir, wf))
        return fL

def getBatchImg(DirList, batchIDs, maskDir, stylDir, extraInfo, device, refMat):
    imgBatch = []
    maskBatch = []
    stylBatch = []
    MatIDBatch = []
    for id in range(len(batchIDs)):
        imgN = DirList[id]
        img = Image.open(imgN)
        img = Img_transform(img)
        img = img.unsqueeze(0)
        imgBatch.append(img)

        fname = os.path.basename(imgN).split("_")[-1]
        stylN = os.path.join(os.path.dirname(imgN), f"hires_{fname}")
        styI = Image.open(stylN)
        styI = Img_transform(styI)
        styI = styI.unsqueeze(0)
        stylBatch.append(styI)

        mask = torch.ones_like(img[:, 0, :, :].unsqueeze(1))
        maskBatch.append(mask)

    imgBatch = torch.cat(imgBatch, dim=0).to(device)
    maskBatch = torch.cat(maskBatch, dim=0).to(device)
    stylBatch = torch.cat(stylBatch, dim=0).to(device)

    return imgBatch, maskBatch, stylBatch, MatIDBatch

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
nm_paths_te = {"in": [], "gt": []}
for folder in ["anger", "fear"]:
    in_paths = sorted(glob.glob(os.path.join(in_dir, folder, "lores*")))
    gt_paths = [os.path.join(in_dir, folder, "higres_{}".format(os.path.basename(p).split("_")[-1])) for p in in_paths]

    nm_paths_te["in"].extend(in_paths)
    nm_paths_te["gt"].extend(gt_paths)
    
    # nm_paths_te["in"].append(in_paths[0])
    # nm_paths_te["gt"].append(gt_paths[0])

test_paths = nm_paths_te["in"]
n_test = len(test_paths)
TestKK = 1 # n_test // BSize
print("test_paths:", len(nm_paths_te["in"]))

show_idList = test_paths    
show_inBatch, show_maskBatch, show_stylBatch, show_matIDs = getBatchImg(DirList=show_idList, batchIDs=show_idList,
                                                                        maskDir=[], stylDir=[],
                                                                        extraInfo=[], device=device,
                                                                        refMat=[])
print(show_inBatch.size(), show_maskBatch.size(), show_stylBatch.size())

#model = Generator_CNN_Cat(3+len(test_InfoDir)*3, 3).to(device)
model = Generator_CNNCIN(inDim=3, outDim=3, styleNum=NumMat).to(device)
model.eval()

lossFunc = BatchFeatureLoss_Model(device=device, c_alpha=1., s_beta=1.e4, s_layWei=[1., 1., 1., 1., 1.]).to(device)

plot_dir = "output_ours/test/plot"
save_dir = "output_ours/test/predictions"
os.makedirs(plot_dir, exist_ok=1)
os.makedirs(save_dir, exist_ok=1)
losses = {"index": [], "c_Loss": [], "s_Loss": []}

def model_test(data_index, fname):
    DList = [test_paths[data_index]]
    test_inBatch, test_maskBatch, test_stylBatch, test_MatIDBatch = \
        getBatchImg(DirList=DList, batchIDs=DList, maskDir=[], stylDir=[],
                    extraInfo=[], device=device, refMat=[])

    model.eval()
    with torch.no_grad():
        path = str(test_paths[data_index]).split("/")
        seq_name = path[-2]

        Out = model(test_inBatch)
        c_Loss, s_Loss = lossFunc(X=Out, SG=test_stylBatch, CX=test_inBatch[:, 0:3, :, :], MX=test_maskBatch)
        Loss = c_Loss + s_Loss
        Loss = Loss

        losses["c_Loss"].append(c_Loss.item())
        losses["s_Loss"].append(s_Loss.item())
        losses["index"].append(data_index)

        # save normal map
        bname = os.path.basename(test_paths[data_index]).split(".png")[0]
        out_fname = f"{data_idx:03d}_{seq_name}_{bname}.png"
        saveN = os.path.join(save_dir, out_fname)
        print("saveN:", saveN)
        rst = Out[0, 0:3, :, :] * torch.tensor(vgg_std).view(-1, 1, 1).to(device) + \
              torch.tensor(vgg_mean).view(-1, 1, 1).to(device)
        save_image(rst, fp=saveN)

        # save plot
        save_path = os.path.join(plot_dir, out_fname)
        save_image_plot(test_inBatch, Out, test_stylBatch, save_path)

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

for data_idx in range(len(test_paths)):
    fname = os.path.basename(test_paths[data_idx])
    print("fname:", fname)
    test_loss = model_test(data_idx, fname)

print("### DONE")