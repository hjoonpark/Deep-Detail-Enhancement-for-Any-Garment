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

def save_image_plot(b, I_ins, I_outs, I_gts, save_path):
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

    plt.suptitle("Iteration: {}, index: {}".format(itt, b))
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

args = parser.parse_args()

#USE_CUDA = False
USE_CUDA = torch.cuda.is_available()
print("balin-->", USE_CUDA)
device = torch.device("cuda:0" if USE_CUDA else "cpu")

Patch_H = 128
Patch_W = 128

NumMat = 5

Img_transform = transforms.Compose([transforms.Resize((Patch_H, Patch_W)), transforms.ToTensor(),
                                    transforms.Normalize(vgg_mean, vgg_std)])
Mask_transform = transforms.Compose([transforms.Resize((Patch_H, Patch_W)), transforms.ToTensor()])


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
    for id in batchIDs:
        imgN = DirList[id]
        p = imgN.split('/')
        maskN = maskDir + p[-1]
        stylN = stylDir + p[-1]
        if len(refMat) > 0:
            pm = imgN.split('_')
            pmm = pm[-1].split('.')
            mid = refMat[int(pmm[0])]
            MatIDBatch.append(mid)

        img = Image.open(imgN)
        img = Img_transform(img)
        img = img.unsqueeze(0)
        Maps = []
        for i in range(len(extraInfo)):
            mapN = extraInfo[i] + p[-1]
            imap = Image.open(mapN)
            #imap = Mask_transform(imap)
            imap = Img_transform(imap)
            imap = imap.unsqueeze(0)
            Maps.append(imap)
        if len(Maps) > 0:
            Maps = torch.cat(Maps, dim=1)
            img = torch.cat([img, Maps], dim=1)
        imgBatch.append(img)

        mask = Image.open(maskN)
        mask = Mask_transform(mask)
        mask = mask.unsqueeze(0)
        maskBatch.append(mask)

        styI = Image.open(stylN)
        styI = Img_transform(styI)
        styI = styI.unsqueeze(0)
        stylBatch.append(styI)

    imgBatch = torch.cat(imgBatch, dim=0).to(device)
    maskBatch = torch.cat(maskBatch, dim=0).to(device)
    stylBatch = torch.cat(stylBatch, dim=0).to(device)

    return imgBatch, maskBatch, stylBatch, MatIDBatch


def saveBImg(DirList, batchIDs, imgBatch, maskBatch, saveDir):
    c = 0
    for id in batchIDs:
        imgN = DirList[id]
        bname = os.path.basename(imgN).split(".png")[0]
        saveN = os.path.join(saveDir, f"{itt:08d}_{bname}.png")

        p = imgN.split('/')
        rst = imgBatch[c, 0:3, :, :] * torch.tensor(vgg_std).view(-1, 1, 1).to(device) + \
              torch.tensor(vgg_mean).view(-1, 1, 1).to(device)
        save_image(rst*maskBatch[c], fp=saveN)
        c = c+1
    return


BSize = 32
BMatIDs = [0 for _ in range(BSize)]
#NumInfoMap = 8

'''
    0: Chamuse_Silk, 1: WoolMelton, 2: Knit_Terry,
    3: Chiffon_Silk, 4: DenimLight
'''
train_RefMat = [0, 0, 0, 1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4]
train_inDir = '../TrainData/MatCC/e__in/'
train_maskDir = '../TrainData/MatCC/e__Mask/'
train_stylDir = '../TrainData/MatCC/e__gt/'
#train_InfoDir = ['../TrainData/TT_InfoPatch/APT_in_b/1/', '../TrainData/TT_InfoPatch/APT_in_a/1/']
#train_InfoDir = ['../Data/case_1/Chamuse/InfoPatch/P_JJ/'+str(i) + '/' for i in range(NumInfoMap)]
train_InfoDir = []

wF = ["*_1.png", "*_2.png", "*_3.png", "*_4.png",
      "*_7.png", "*_8.png", "*_9.png", "*_10.png", "*_11.png", "*_12.png"]
train_imgList = get_fileSet_list(train_inDir, withF=wF)

train_idList = [i for i in range(len(train_imgList))]
numTrainD = len(train_imgList)
TrainKK = numTrainD // BSize
print(numTrainD)
print(TrainKK)

'''
    0: Chamuse_Silk, 1: WoolMelton, 2: Knit_Terry
'''
test_RefMat = [0, 1, 2]
test_inDir = '../TrainData/MatCC/e__in/'
test_maskDir = '../TrainData/MatCC/e__Mask/'
test_stylDir = '../TrainData/MatCC/e__gt/'
#test_InfoDir = ['../TrainData/TT_InfoPatch/PE_in_b/1/', '../TrainData/TT_InfoPatch/PE_in_a/1/']
#test_InfoDir = ['../Data/case_1/Chamuse/InfoPatch/T_JJ/'+str(i) + '/' for i in range(NumInfoMap)]
test_InfoDir = []

test_imgList = get_fileSet_list(dir=test_inDir, withF=[])
test_idList = [i for i in range(len(test_imgList))]
numTestD = len(test_imgList)
TestKK = numTestD // BSize
print(numTestD)

show_idList = test_idList[0:BSize]
show_inBatch, show_maskBatch, show_stylBatch, show_matIDs = getBatchImg(DirList=test_imgList, batchIDs=show_idList,
                                                                        maskDir=test_maskDir, stylDir=test_stylDir,
                                                                        extraInfo=test_InfoDir, device=device,
                                                                        refMat=test_RefMat)
print(show_inBatch.size(), show_maskBatch.size(), show_stylBatch.size())
print(show_matIDs)

# saveBImg(test_imgList, show_idList, show_inBatch, show_maskBatch, saveDir='../PatchTest/in_')
# saveBImg(test_imgList, show_idList, show_stylBatch, show_maskBatch, saveDir='../PatchTest/gt_')

shuffle(train_idList)
shuffle(test_idList)

#model = Generator_CNN_Cat(3+len(test_InfoDir)*3, 3).to(device)
model = Generator_CNNCIN(inDim=3, outDim=3, styleNum=NumMat).to(device)
print(model)
n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Trainable parameters={:,}".format(n_trainable_params))

lossFunc = BatchFeatureLoss_Model(device=device, c_alpha=1., s_beta=1.e4, s_layWei=[1., 1., 1., 1., 1.]).to(device)

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=True)

plot_dir = "output/log"
os.makedirs(plot_dir, exist_ok=1)
losses = {"iter": [], "c_Loss": [], "s_Loss": []}
def model_train(itt):
    t = time.time()
    k = itt % TrainKK
    if k == 0:
        shuffle(train_idList)

    DList = train_idList[k * BSize: (k + 1) * BSize]
    train_inBatch, train_maskBatch, train_stylBatch, train_MatIDBatch = \
        getBatchImg(DirList=train_imgList, batchIDs=DList, maskDir=train_maskDir, stylDir=train_stylDir,
                    extraInfo=train_InfoDir, device=device, refMat=train_RefMat)

    model.train()
    optimizer.zero_grad()

    Out = model(train_inBatch, train_MatIDBatch)
    c_Loss, s_Loss = lossFunc(X=Out, SG=train_stylBatch, CX=train_inBatch[:, 0:3, :, :], MX=train_maskBatch)
    Loss = c_Loss + s_Loss
    Loss = Loss

    Loss.backward()
    optimizer.step()
    if itt % 50 == 0:
        print("Train Iter: {}, Content Loss: {:.3f}, Style Loss: {:.3f}, Loss: {:.3f}, time:{:.4f}s".
            format(itt, c_Loss.item(), s_Loss.item(), Loss.item(), time.time() - t))
        
        losses["c_Loss"].append(c_Loss.item())
        losses["s_Loss"].append(s_Loss.item())
        losses["iter"].append(itt)

    if itt % 500 == 0:
        plt.figure(figsize=(10, 5))
        plt.plot(losses['iter'], losses["c_Loss"], label="c_Loss")
        plt.plot(losses['iter'], losses["s_Loss"], label="s_Loss")
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend()
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.yscale("log")
        plt.title("Iteration: {}".format(itt))
        plt.tight_layout()
        plt.grid()
        save_path = os.path.join(plot_dir, "losses.png")
        plt.savefig(save_path, dpi=150)
        print("loss saved:", save_path)

        b = 0
        save_path = os.path.join(plot_dir, "train_image.png")
        save_image_plot(b, train_inBatch, Out, train_stylBatch, save_path)
    return Loss


def model_test(itt):
    t = time.time()
    k = itt % TestKK
    if k == 0:
        shuffle(test_idList)

    DList = test_idList[k * BSize: (k + 1) * BSize]
    test_inBatch, test_maskBatch, test_stylBatch, test_MatIDBatch = \
        getBatchImg(DirList=test_imgList, batchIDs=DList, maskDir=test_maskDir, stylDir=test_stylDir,
                    extraInfo=test_InfoDir, device=device, refMat=test_RefMat)

    model.eval()
    with torch.no_grad():
        Out = model(test_inBatch, test_MatIDBatch)
        c_Loss, s_Loss = lossFunc(X=Out, SG=test_stylBatch, CX=test_inBatch[:, 0:3, :, :], MX=test_maskBatch)
        Loss = c_Loss + s_Loss
        Loss = Loss

        print("Test Iter: {}, Content Loss: {:.3f}, Style Loss: {:.3f}, Loss: {:.3f}, time:{:.4f}s".
              format(itt, c_Loss.item(), s_Loss.item(), Loss.item(), time.time() - t))

        b = 0
        save_path = os.path.join(plot_dir, "test_image.png")
        save_image_plot(b, test_inBatch, Out, test_stylBatch, save_path)

        return Loss


def rst_show(itt):
    model.eval()
    with torch.no_grad():
        Out = model(show_inBatch, show_matIDs)
        return Out


#iterations = args.epochs * TrainKK + 1
iterations = 550000
t_total = time.time()
loss_values = []
bad_counter = 0
best = 1.e16
best_iter = 0
invSave = 0

IFTRAIN = True
betit = -1

# ckp = torch.load('./ckp/DDE_model.ckp', map_location=lambda storage, loc: storage)
# model.load_state_dict(ckp['model'])
# optimizer.load_state_dict(ckp['optimizer'])
# betit = ckp['itter']
# best = ckp['loss']
# loss_values.append(best)
# print("begLoss: " + str(best) + " >> begIter: " + str(betit))

if IFTRAIN:
    IFSumWriter = False
    save_dir = "output/PatchTest"
    os.makedirs(save_dir, exist_ok=1)
    ckp_dir = os.path.join(save_dir, "..", "ckp")
    os.makedirs(ckp_dir, exist_ok=1)

    if IFSumWriter:
        from torch.utils.tensorboard import SummaryWriter
        #from tensorboardX import SummaryWriter
        writer = SummaryWriter()

    for itt in range(betit + 1, iterations+1):
        train_loss = model_train(itt)

        if itt % 100 == 0 or itt == iterations - 1:
            test_loss = model_test(itt)
            if IFSumWriter:
                writer.add_scalar('testLoss', test_loss, itt)
                writer.add_scalar('trainLoss', train_loss, itt)

            if itt % 500 == 0 or itt == iterations - 1:
                Out = rst_show(itt)
                saveBImg(test_imgList, show_idList, Out, show_maskBatch, saveDir=save_dir)

                if itt % 50000 == 0:
                    saveName = os.path.join(ckp_dir, 't_model_' + str(itt) + '.ckp')
                elif itt % 1000 == 0:
                    saveName = os.path.join(ckp_dir, 'c_model_1000.ckp')
                else:
                    saveName = os.path.join(ckp_dir, 'c_model_500.ckp')
                loss_values.append(test_loss.item())
                torch.save({'itter': itt, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                            'loss': test_loss}, saveName)

                if loss_values[-1] < best:
                    best = loss_values[-1]
                    best_iter = itt
                    invSave = 0
                    bad_counter = 0
                    shutil.copyfile(saveName, os.path.join(ckp_dir, 'c_best_model.ckp'))
                else:
                    bad_counter += 500

                if bad_counter == args.patience * TrainKK:
                    break

    if IFSumWriter:
        writer.close()

