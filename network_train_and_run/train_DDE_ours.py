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

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--patience', type=int, default=100000, help='Patience')

args = parser.parse_args()

#USE_CUDA = False
USE_CUDA = torch.cuda.is_available()
print("balin-->", USE_CUDA)
device = torch.device("cuda:0" if USE_CUDA else "cpu")

Patch_H = 256
Patch_W = 256

NumMat = 5

Img_transform = transforms.Compose([transforms.ToTensor(),
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


def saveBImg(DirList, batchIDs, imgBatch, maskBatch, saveDir):
    c = 0
    for id in range(len(batchIDs)):
        imgN = DirList[id]
        p = imgN.split('/')
        saveN = saveDir + p[-1]
        # rst = imgBatch[c, 0:3, :, :] * torch.tensor(vgg_std).view(-1, 1, 1).to(device) + \
            #   torch.tensor(vgg_mean).view(-1, 1, 1).to(device)
        rst = imgBatch[c, 0:3, :, :]
        save_image(rst*maskBatch[c], fp=saveN)

        if imgBatch.shape[0] < c-1:
            c = c+1
        else:
            break
    return

"""
Load train dataset
"""
in_dir = "nvidia_data/BakedUVMaps"
nm_paths = {"in": [], "gt": []}
for folder in ["amazement", "pain"]:
    in_paths = sorted(glob.glob(os.path.join(in_dir, folder, "lores*")))
    gt_paths = [os.path.join(in_dir, folder, "higres_{}".format(os.path.basename(p).split("_")[-1])) for p in in_paths]

    nm_paths["in"].extend(in_paths)
    nm_paths["gt"].extend(gt_paths)

BSize = 2
train_paths = nm_paths["in"]
n_train = len(train_paths)
TrainKK = n_train // BSize

"""
Load test dataset
"""
nm_paths_te = {"in": [], "gt": []}
for folder in ["anger", "fear"]:
    in_paths = sorted(glob.glob(os.path.join(in_dir, folder, "lores*")))
    gt_paths = [os.path.join(in_dir, folder, "higres_{}".format(os.path.basename(p).split("_")[-1])) for p in in_paths]

    nm_paths_te["in"].extend(in_paths)
    nm_paths_te["gt"].extend(gt_paths)

test_paths = nm_paths_te["in"]
n_test = len(test_paths)
TestKK = 100#n_test // BSize

show_idList = test_paths[0:BSize]
show_inBatch, show_maskBatch, show_stylBatch, show_matIDs = getBatchImg(DirList=show_idList, batchIDs=show_idList,
                                                                        maskDir=[], stylDir=[],
                                                                        extraInfo=[], device=device,
                                                                        refMat=[])
print(show_inBatch.size(), show_maskBatch.size(), show_stylBatch.size())
print(show_matIDs)

#model = Generator_CNN_Cat(3+len(test_InfoDir)*3, 3).to(device)
model = Generator_CNNCIN(inDim=3, outDim=3, styleNum=NumMat).to(device)
print(model)

lossFunc = BatchFeatureLoss_Model(device=device, c_alpha=1., s_beta=1.e4, s_layWei=[1., 1., 1., 1., 1.]).to(device)

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=True)

def model_train(itt):
    t = time.time()
    k = itt % TrainKK
    if k == 0:
        shuffle(train_paths)

    DList = train_paths[k * BSize: (k + 1) * BSize]
    train_inBatch, train_maskBatch, train_stylBatch, train_MatIDBatch = \
        getBatchImg(DirList=DList, batchIDs=DList, maskDir=[], stylDir=[],
                    extraInfo=[], device=device, refMat=[])

    model.train()
    optimizer.zero_grad()

    Out = model(train_inBatch)
    c_Loss, s_Loss = lossFunc(X=Out, SG=train_stylBatch, CX=train_inBatch[:, 0:3, :, :], MX=train_maskBatch)
    Loss = c_Loss + s_Loss
    Loss = Loss

    Loss.backward()
    optimizer.step()
    if itt % 100 == 0:
        print("Train Iter: {}, Content Loss: {:.3f}, Style Loss: {:.3f}, Loss: {:.3f}, time:{:.4f}s".
            format(itt, c_Loss.item(), s_Loss.item(), Loss.item(), time.time() - t))
    return Loss

def model_test(itt):
    TestKK = 1
    t = time.time()
    k = itt % TestKK
    if k == 0:
        shuffle(test_paths)

    DList = test_paths[k * BSize: (k + 1) * BSize]
    test_inBatch, test_maskBatch, test_stylBatch, test_MatIDBatch = \
        getBatchImg(DirList=DList, batchIDs=DList, maskDir=[], stylDir=[],
                    extraInfo=[], device=device, refMat=[])

    model.eval()
    with torch.no_grad():
        Out = model(test_inBatch)
        c_Loss, s_Loss = lossFunc(X=Out, SG=test_stylBatch, CX=test_inBatch[:, 0:3, :, :], MX=test_maskBatch)
        Loss = c_Loss + s_Loss
        Loss = Loss

        print("Test Iter: {}, Content Loss: {:.3f}, Style Loss: {:.3f}, Loss: {:.3f}, time:{:.4f}s".
              format(itt, c_Loss.item(), s_Loss.item(), Loss.item(), time.time() - t))

        return Loss


def rst_show(itt):
    model.eval()
    with torch.no_grad():
        Out = model(show_inBatch)
        return Out


#iterations = args.epochs * TrainKK + 1
iterations = 55000
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

        if itt % 1000 == 0 or itt == iterations - 1:
            test_loss = model_test(itt)
            if IFSumWriter:
                writer.add_scalar('testLoss', test_loss, itt)
                writer.add_scalar('trainLoss', train_loss, itt)

            if itt % 1000 == 0 or itt == iterations - 1:
                Out = rst_show(itt)
                
                saveBImg(show_idList, show_idList, Out, show_maskBatch, saveDir=save_dir)
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

