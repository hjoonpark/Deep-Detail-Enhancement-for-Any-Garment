from __future__ import division
from __future__ import print_function

import time
import argparse
import os
import glob
import torch
import shutil, sys
from torchvision import transforms
from PIL import Image
from random import shuffle
import torch.optim as optim

# from models import Generator_CNNCIN
from models import Generator_CNNCIN_nvidia as Generator_CNNCIN
from VGG_Model import vgg_mean, vgg_std
from Losses import BatchFeatureLoss_Model
from torchvision.utils import save_image

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import numpy as np
from dataset import DatasetNormalmaps
import torch.multiprocessing as mp
from ddp import DDPWraper
from gpu import GPUStat

def stat(x):
    out_str = "{}, min/max=({:.2f}, {:.2f}), mean={:.2f}, std={:.2f}".format(x.shape, x.min(), x.max(), x.mean(), x.std())
    return out_str
def set_all_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_image_plot(itt, b, I_ins, I_outs, I_gts, save_path):
    std = np.array(vgg_std)
    mean = np.array(vgg_mean)

    fig = plt.figure(figsize=(15, 11))
    ax = fig.add_subplot(231)
    ax.set_xlabel("Input low-res")
    I_in = I_ins[b].cpu().permute(1, 2, 0).numpy()
    I_in = I_in*std + mean
    I_in[I_in<0], I_in[I_in>1] = 0, 1
    ax.imshow(I_in)
    ax.set_title("{} ({:.2f}, {:.2f}) [{:.2f}, {:.2f}]".format(I_in.shape, I_in.min(), I_in.max(), I_in.mean(), I_in.std()))

    ax = fig.add_subplot(232)
    ax.set_xlabel("Reconstruction high-res")
    I_out = I_outs[b].cpu().permute(1, 2, 0).detach().numpy()
    I_out = I_out*std + mean
    I_out[I_out<0], I_out[I_out>1] = 0, 1
    # I_out = (I_out-I_out.min())/(I_out.max()-I_out.min())
    ax.imshow(I_out)
    ax.set_title("{} ({:.2f}, {:.2f}) [{:.2f}, {:.2f}]".format(I_out.shape, I_out.min(), I_out.max(), I_out.mean(), I_out.std()))
    
    ax = fig.add_subplot(233)
    ax.set_xlabel("Target high-res")
    I_gt = I_gts[b].cpu().permute(1, 2, 0).detach().numpy()
    I_gt = I_gt*std + mean
    I_gt[I_gt<0], I_gt[I_gt>1] = 0, 1
    # I_gt = (I_gt-I_gt.min())/(I_gt.max()-I_gt.min())
    ax.imshow(I_gt)
    ax.set_title("{} ({:.2f}, {:.2f}) [{:.2f}, {:.2f}]".format(I_gt.shape, I_gt.min(), I_gt.max(), I_gt.mean(), I_gt.std()))

    L_diff = I_out-I_in
    H_diff = I_out-I_gt
    HL_diff = I_gt-I_in
    L_diff[L_diff<0], L_diff[L_diff>1] = 0, 1
    H_diff[H_diff<0], H_diff[H_diff>1] = 0, 1
    HL_diff[HL_diff<0], HL_diff[HL_diff>1] = 0, 1
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

    plt.suptitle("Epoch: {}, index: {}".format(itt, b))
    plt.tight_layout()
    # save_path = os.path.join(plot_dir, "test_image.png")
    plt.savefig(save_path, dpi=150)
    print("image plot saved:", save_path)
 
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
   

def rst_show(itt):
    model.eval()
    with torch.no_grad():
        Out = model(show_inBatch)
        return Out

def main(rank, world_size, args):
    device = torch.device(f"cuda:{rank}")
    gpu_stat = GPUStat()
    batch_size = args.batch_size
    log_epoch = args.log_epoch
    ckp_epoch = args.ckp_epoch

    dataset_tr = DatasetNormalmaps(is_train=True)
    if rank == 0:
        dataset_te = DatasetNormalmaps(is_train=False)
        data_loader_te = torch.utils.data.DataLoader(dataset_te, batch_size=batch_size)
    
    NumMat = 1
    model = Generator_CNNCIN(inDim=3, outDim=3, styleNum=NumMat).to(device)
    # model = torch.compile(model)
    print(model)
    n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable parameters={:,}".format(n_trainable_params))

    if world_size > 1:
        ddp = DDPWraper(rank=rank, world_size=world_size)
        model = ddp.setup_model(model)
        data_loader_tr = ddp.setup_dataloader(dataset_tr, batch_size)
        model_module = model.module
    else:
        data_loader_tr = torch.utils.data.DataLoader(dataset_tr, batch_size=batch_size)
        model = model.to(device)
        model_module = model

    Patch_H = 512
    Patch_W = 512

    NumMat = 1

    """
    Load train dataset
    """
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=True)
    lossFunc = BatchFeatureLoss_Model(device=device, c_alpha=1., s_beta=1.e4, s_layWei=[1., 1., 1., 1., 1.]).to(device)

    #iterations = args.epochs * TrainKK + 1
    iterations = int(args.epochs)
    t_total = time.time()
    loss_values = []
    bad_counter = 0
    best = 1.e16
    best_iter = 0
    invSave = 0

    betit = -1

    if args.model_path != "":
        model_path = args.model_path
        try:
            ckp = torch.load(args.model_path, map_location=lambda storage, loc: storage)
            model.load_state_dict(ckp['model'])
            # optimizer.load_state_dict(ckp['optimizer'])
            betit = ckp['itter']
            best = ckp['test_loss']
            loss_values.append(best)
            print(":: loading model: {}".format(model_path))
            print("  loaded succesfully!")
            print("begLoss: " + str(best) + " >> begIter: " + str(betit))
        except Exception as e:
            print(":: failed to load {} - architecture mismatch! Initializing new instead".format(model_path))
            print(e)
        
    if rank == 0:
        save_dir = "output_ours"
        ckp_dir = os.path.join(save_dir, "ckp")
        os.makedirs(ckp_dir, exist_ok=1)
        plot_dir = "output_ours/log"
        os.makedirs(plot_dir, exist_ok=1)
        losses = {"iter": [], "c_Loss": [], "s_Loss": []}

        print(args)
        sys.stdout.flush()

    def model_test(itt, batch_idx, epoch, data, device):
        t = time.time()
        model.eval()
        with torch.no_grad():
            lres = data["lres"].to(device)
            hres = data["hres"].to(device)
            mask = data["mask"].to(device)
            Out = model(lres)
            c_Loss, s_Loss = lossFunc(X=Out, SG=hres, CX=hres[:, 0:3, :, :], MX=mask)
            Loss = c_Loss + s_Loss

            if batch_idx == 0:
                print("Test epoch: {}, Content Loss: {:.3f}, Style Loss: {:.3f}, Loss: {:.3f}, time:{:.4f}s".
                    format(epoch, c_Loss.item(), s_Loss.item(), Loss.item(), time.time() - t))

                b = 0
                save_path = os.path.join(plot_dir, "test_image.png")
                save_image_plot(epoch, b, lres, Out, hres, save_path)
                
            return Loss
        
    def model_train(model, batch_idx, epoch, data, device):
        t = time.time()
        lres = data["lres"].to(device)
        hres = data["hres"].to(device)
        mask = data["mask"].to(device)
        Out = model(lres)
        c_Loss, s_Loss = lossFunc(X=Out, SG=hres, CX=hres[:, 0:3, :, :], MX=mask)
        Loss = c_Loss + s_Loss
        Loss = Loss

        Loss.backward()

        if rank == 0 and batch_idx == 0:
            losses["c_Loss"].append(c_Loss.item())
            losses["s_Loss"].append(s_Loss.item())
            losses["iter"].append(epoch)

            if epoch % log_epoch == 0:
                print("Train epoch: {}, Content Loss: {:.3f}, Style Loss: {:.3f}, Loss: {:.3f}, time:{:.4f}s".
                    format(epoch, c_Loss.item(), s_Loss.item(), Loss.item(), time.time() - t))
                
            if epoch % ckp_epoch == 0:
                plt.figure(figsize=(10, 5))
                plt.plot(losses['iter'], losses["c_Loss"], label="c_Loss")
                plt.plot(losses['iter'], losses["s_Loss"], label="s_Loss")
                plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
                plt.legend()
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.yscale("log")
                plt.title("Epoch: {} | c_Loss={:.4f}, s_Loss={:.4f}".format(epoch, losses["c_Loss"][-1], losses["s_Loss"][-1]))
                plt.tight_layout()
                plt.grid()
                save_path = os.path.join(plot_dir, "losses.png")
                plt.savefig(save_path, dpi=150)
                print("loss saved:", save_path)

                b = 0
                save_path = os.path.join(plot_dir, "train_image.png")
                save_image_plot(epoch, b, lres, Out, hres, save_path)

                print(gpu_stat.get_stat_str())
        return Loss
    sys.stdout.flush()

    for itt in range(betit + 1, iterations+1):
        if world_size > 1:
            data_loader_tr.sampler.set_epoch(itt)

        model.train()
        for batch_idx, data_tr in enumerate(data_loader_tr):
            optimizer.zero_grad()
            train_loss = model_train(model_module, batch_idx, itt, data_tr, device)
            optimizer.step()
            sys.stdout.flush()

        if rank != 0:
            continue

        if itt > 0 and ((itt % ckp_epoch == 0 or itt == iterations - 1) or itt == 10):
            test_losses = []
            for batch_idx, data_te in enumerate(data_loader_te):
                test_loss_i = model_test(model_module, batch_idx, itt, data_te, device)
                test_losses.append(test_loss_i.item()/len(data_te))
                sys.stdout.flush()
            test_loss = sum(test_losses) / len(test_losses)

            saveName = os.path.join(ckp_dir, 't_model.ckp')
            loss_values.append(test_loss)
            torch.save({'itter': itt, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'test_loss': test_loss}, saveName)

            if loss_values[-1] < best:
                best = loss_values[-1]
                best_iter = itt
                invSave = 0
                to_path = os.path.join(ckp_dir, f'c_best_model_{best_iter}.ckp')
                shutil.copyfile(saveName, to_path)
                print("Best model saved (loss={:.3f}): {}".format(best, to_path))
            sys.stdout.flush()
            
    print("### DONE ###")

if __name__ == "__main__":
    set_all_seeds(0)

    # Training settings
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--world-size", type=int, help="number of GPUs", default=1)
    parser.add_argument('--epochs', type=int, default=9e5, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=1e-5, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--patience', type=int, default=100000, help='Patience')
    parser.add_argument('--batch-size', type=int, default=20, help='batch size')
    parser.add_argument('--ckp-epoch', type=int, default=50, help='checkpoint epoch')
    parser.add_argument('--log-epoch', type=int, default=5, help='log epoch')
    parser.add_argument('--model-path', type=str, default="")
    args = parser.parse_args()
    world_size = args.world_size
    print(f">> args: {args}")
    
    if world_size > 1:
        mp.spawn(main, args=[world_size, args], nprocs=world_size)
    else:
        main(0, 1, args) # for debugging

