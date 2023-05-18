from DataIO import *
import torch
import numpy as np
from torch import optim
import os, math
import torch.nn as nn
from Losses import Loss_NormalCrossVert, LapSmooth_loss
import time
import cv2, glob, sys

# import networkx as nx
# import plotly
# import plotly.graph_objects as go

import argparse, sys, datetime

#USE_CUDA = False
USE_CUDA = torch.cuda.is_available()
print('balin-->', USE_CUDA)
# num_iter = 10000
# cEdgeWeight = 1
# distWeight = 0.1
# smoothWeight = 0.01
num_iter = 2000

# v1
# cEdgeWeight = 100.
# distWeight = 1
# smoothWeight = 0.01

cEdgeWeight = 200.
distWeight = 0.1
smoothWeight = 0.0001

def get_timestamp():
    now = datetime.datetime.now()
    timestamp = "{:02}-{:02}-{:02} {:02}:{:02}:{:02}".format(now.year, now.month, now.day, now.hour, now.minute, now.second)
    return timestamp
    
def save_plotly(iteration, save_path, x, rrNormArray, vertEdges_0):
    L = 1.5
    colors = rrNormArray.copy()

    x_edges1, y_edges1, z_edges1 = [], [], []
    for v_idx, normal in enumerate(rrNormArray):
        src = x[v_idx].copy()
        normal2 = normal

        trg = src + L*normal2

        es = [(src, trg)]
        for e in es:
            x_edges1 += (e[0][0], e[1][0], None)
            y_edges1 += (e[0][1], e[1][1], None)
            z_edges1 += (e[0][2], e[1][2], None)

    # x_edges2, y_edges2, z_edges2 = [], [], []
    # for i in range(len(vertEdges_0)):
    #     a = vertEdges_0[i]
    #     b = vertEdges_1[i]
    #     x_edges2 += (x[a][0], x[b][0], None)
    #     y_edges2 += (x[a][1], x[b][1], None)
    #     z_edges2 += (x[a][2], x[b][2], None)

    #create a trace for the nodes
    v = x
    nodes1 = go.Scatter3d(
        x=v[:,0],
        y=v[:,1],
        z=v[:,2],
        mode='markers',
        marker=dict(symbol='circle',
                size=1,
                color=colors)
        )
    edges1 = go.Scatter3d(
        x=x_edges1,
        y=y_edges1,
        z=z_edges1,
        mode='lines',
        line=dict(color=colors, width=0.3),
        hoverinfo='none'
    )
    # edges2 = go.Scatter3d(
    #     x=x_edges2,
    #     y=y_edges2,
    #     z=z_edges2,
    #     mode='lines',
    #     line=dict(color="black", width=1),
    #     hoverinfo='none'
    # )
    s = 1.2
    mins = (x.min(0)+L)*s
    maxs = (x.max(0)+L)*s
    G_data2 = [nodes1, edges1]
    fig = go.Figure(data=G_data2)
    fig.update_layout(autosize=False, width=960, height=760, )
    camera = dict(
        eye=dict(x=0.1, y=-0.1, z=2)
    )
    fig.update_layout(
        scene_camera=camera,
        title="iter: {}".format(iteration),
        scene = dict(
            xaxis = dict(range=[mins[0], maxs[0]],),
            yaxis = dict(range=[mins[1], maxs[1]],),
            zaxis = dict(range=[mins[2], maxs[2]]))
    )
    fig.write_image(save_path) 

def write_obj(save_path, points, normals=[], faces=[], vts=[]):
    """
    x: (N, 3)
    faces: (M, 3) - triangle faces
    """
    with open(save_path, "w+") as file:
        for i, v in enumerate(points):
            file.write("v {} {} {}\n".format(v[0], v[1], v[2]))
            
        for i, vn in enumerate(normals):
            file.write("vn {} {} {}\n".format(vn[0], vn[1], vn[2]))
            
        for i, vt in enumerate(vts):
            file.write("vt {} {}\n".format(vt[0], vt[1]))
            
        for i, f in enumerate(faces):
            if len(f) == 3:
                # i = face index
                # f: 3d vector
                f1 = f[0] + 1
                f2 = f[1] + 1
                f3 = f[2] + 1
                file.write("f {}//{} {}//{} {}//{}\n".format(f1, f1, f2, f2, f3, f3))
            elif len(f) == 4:
                # i = face index
                # f: 3d vector
                f1 = f[0] + 1
                f2 = f[1] + 1
                f3 = f[2] + 1
                file.write("f {}//{} {}//{} {}//{}\n".format(f1, f1, f2, f2, f3, f3))
                
                # f: 3d vector
                f1 = f[0] + 1
                f2 = f[2] + 1
                f3 = f[3] + 1
                file.write("f {}//{} {}//{} {}//{}\n".format(f1, f1, f2, f2, f3, f3))
            else:
                assert 0
def read_obj(path):
    faces = []
    hv = []
    hvn = []
    with open(path, "r") as f:
        lines = f.readlines()
        for line in lines:
            ls = line.split(" ")
            if ls[0].lower() == "f":
                f1 = int(ls[1].split("//")[0])
                f2 = int(ls[2].split("//")[0])
                f3 = int(ls[3].split("//")[0])
                faces.append([f1, f2, f3])
            elif ls[0].lower() == "v":
                v1 = float(ls[1])
                v2 = float(ls[2])
                v3 = float(ls[3])
                hv.append([v1, v2, v3])
            elif ls[0].lower() == "vn":
                vn1 = float(ls[1])
                vn2 = float(ls[2])
                vn3 = float(ls[3])
                hvn.append([vn1, vn2, vn3])
                
    hv = np.array(hv)
    hvn = np.array(hvn)
    faces = np.array(faces)-1
    return hv, hvn, faces

def texel_interpolation(texture_map, uvs):
    h, w, _ = texture_map.shape

    output = np.zeros((uvs.shape[0], 3)).astype(np.float32)

    for v_idx, uv in enumerate(uvs):
        u, v = uv[0], uv[1]

        # Convert UV coordinates to pixel coordinates
        pixel_x = int(u * texture_map.shape[1])
        pixel_y = int(v * texture_map.shape[0])

        # Determine the four nearest pixels to the UV coordinates
        pixel_top_left = (pixel_x // 1, pixel_y // 1)
        pixel_top_right = (pixel_top_left[0] + 1, pixel_top_left[1])
        pixel_bottom_left = (pixel_top_left[0], pixel_top_left[1] + 1)
        pixel_bottom_right = (pixel_top_left[0] + 1, pixel_top_left[1] + 1)

        # Calculate the color value at the given UV coordinates using bilinear interpolation
        color_top_left = texture_map[pixel_top_left[1], pixel_top_left[0]]
        color_top_right = texture_map[pixel_top_right[1], pixel_top_right[0]]
        color_bottom_left = texture_map[pixel_bottom_left[1], pixel_bottom_left[0]]
        color_bottom_right = texture_map[pixel_bottom_right[1], pixel_bottom_right[0]]

        weight_top_left = (1 - (pixel_x % 1)) * (1 - (pixel_y % 1))
        weight_top_right = (pixel_x % 1) * (1 - (pixel_y % 1))
        weight_bottom_left = (1 - (pixel_x % 1)) * (pixel_y % 1)
        weight_bottom_right = (pixel_x % 1) * (pixel_y % 1)

        output[v_idx, :] = (color_top_left * weight_top_left
                        + color_top_right * weight_top_right
                        + color_bottom_left * weight_bottom_left
                        + color_bottom_right * weight_bottom_right)

    output = output/255.0
    # output = (output-output.min())/(output.max()-output.min())
    output = (output*2)-1
    return output

def geo_opt_ours(fname, device, nmap_path, rrVertArray, uvs, vertEdges_0, vertEdges_1, EdgeCounts, numV, LapM, out_dir):
    texture_map = np.array(cv2.cvtColor(cv2.imread(nmap_path), cv2.COLOR_BGR2RGB))
    # texture_map = np.load(nmap_path).transpose(1, 2, 0)
    rrNormArray = texel_interpolation(texture_map, uvs)
    rrNormTensor = torch.from_numpy(rrNormArray).type(torch.FloatTensor).to(device)
    print(rrNormTensor.min().item(), rrNormTensor.max().item())
    # ini vert position
    noise = torch.from_numpy(rrVertArray.copy()).type(torch.FloatTensor).to(device)
    noise.requires_grad = True

    gtVertTensor = noise.detach().clone()
    
    Func_lossNormalCrossVert = Loss_NormalCrossVert(vertEdges_0, vertEdges_1, EdgeCounts, numV, device).to(device)
    Func_lossVertToGtVert = nn.L1Loss(reduction='mean').to(device)

    adam = optim.Adam(params=[noise], lr=0.001, betas=(0.9, 0.999), amsgrad=True)
    oldLoss = 0.
    t = time.time()
    
    recon_dir = os.path.join(out_dir, "reconstruction")
    os.makedirs(recon_dir, exist_ok=1)
    for iteration in range(num_iter + 1):
        adam.zero_grad()
        loss_geo = Func_lossNormalCrossVert(normalArray=rrNormTensor, vertArray=noise)
        loss_dist = Func_lossVertToGtVert(gtVertTensor, noise)

        total_loss = cEdgeWeight * loss_geo + distWeight * loss_dist
        if LapM is not None:
            loss_smooth = LapSmooth_loss(LapM, noise)
            total_loss += smoothWeight * loss_smooth
        else:
            assert False
            loss_smooth = 0

        derror = abs(oldLoss-total_loss.item())
        if iteration % 10 == 0:
            if not os.path.exists(recon_dir):
                print("EXIT: {} does not exist".format(recon_dir))
                sys.exit()
        if iteration % 20 == 0:
            print("{} [{}] Iteration: {}, derror: {:.5f}, total Loss: {:.5f}, Geo Loss: {:.5f}, disLoss: {:.5f}, Smooth Loss: {:.5f}".format(\
                get_timestamp(), fname, iteration, derror,  total_loss.item(), cEdgeWeight * loss_geo.item(), distWeight * loss_dist, smoothWeight * loss_smooth))
            sys.stdout.flush()
        if derror < 0.001 and iteration > 50:
        # if loss_geo < 1e-3:
            print("{} [{}] Iteration: {}, break : {:.5f}, total Loss: {:.5f}, Geo Loss: {:.5f}, disLoss: {:.5f}, Smooth Loss: {:.5f}".format(\
                get_timestamp(), fname, iteration, derror,  total_loss.item(), cEdgeWeight * loss_geo.item(), distWeight * loss_dist, smoothWeight * loss_smooth))
            sys.stdout.flush()
            return noise.clone().detach()

        oldLoss = total_loss.item()

        # backprop
        total_loss.backward()
        # update parameters
        adam.step()
    return noise.clone().detach()

def run(args):
    folder = args.folder
    idx0 = args.start_idx
    idx1 = args.end_idx + 1
    device = torch.device("cuda:{}".format(args.cuda_idx))

    in_dir = "nvidia_data"
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=1)

    hrestshape, _, hfaces = read_obj(os.path.join(in_dir, "restshape_surf_v1.4.obj"))

    uvs = np.loadtxt(os.path.join(in_dir, "restshape_surf_v1.4_sts.txt"))
    uvs[:,1] = 1-uvs[:,1]

    edges = np.loadtxt(os.path.join(in_dir, "edges_src_trg.txt")).astype(int)
    vertEdges_0 = edges[:,0] # p
    vertEdges_1 = edges[:,1] # q
    EdgeCounts = np.loadtxt(os.path.join(in_dir, "edges_counts.txt")).astype(int)
    numV = len(hrestshape)

    Adj = np.zeros((len(hrestshape), len(hrestshape))).astype(float)
    Adj[vertEdges_0, vertEdges_1] = 1
    Adj[vertEdges_1, vertEdges_0] = 1
    LapM = torch.from_numpy(getLaplacianMatrix(Adj)).float().to(device)

    # Predictions nm
    # nm_dir = "output_ours/test/predictions"
    # hres_paths = sorted(glob.glob(os.path.join(nm_dir, f"*{folder}*")))
    # hres_paths = hres_paths[idx0:min(len(hres_paths), idx1)]

    # Ground-truth nm
    # nm_dir = os.path.join(in_dir, "BakedUVMaps", folder)
    nm_dir = os.path.join(out_dir, "predictions_img")
    hres_paths = sorted(glob.glob(os.path.join(nm_dir, f"*{folder}*.png")))
    if idx1 > idx0:
        hres_paths = hres_paths[idx0:min(len(hres_paths), idx1)]

    print("  ", folder, ":", len(hres_paths), "frames")
    for hres_path in hres_paths:
        # bnames = os.path.basename(hres_path).split("_")
        # seq_name = bnames[1]
        # seq_frame = int(bnames[-1].split(".")[-2])

        # bnames2 = hres_path.split("/")
        # seq_name = bnames2[2]
        # seq_frame = int(bnames2[-1].split(".")[-2])-1

        bnames2 = os.path.basename(hres_path).split("_")
        seq_name = bnames2[0]
        seq_frame = int(bnames2[-1].split(".")[-2])-1

        bname = f"{seq_name}_{seq_frame:03d}"
        save_path = os.path.join(out_dir, "reconstruction", bname)
        if os.path.exists(save_path + ".npy"):
            print("skipping existing: {}".format(save_path))
            continue

        obj_path = os.path.join(in_dir, "obj_7_4", f"{seq_name}_{seq_frame:03d}.obj")
        if not os.path.exists(obj_path):
            print("NOT FOUND:", obj_path)
            assert 0
        x0, _, _ = read_obj(obj_path)
        

        p = geo_opt_ours(bname, device, hres_path, x0, uvs, vertEdges_0, vertEdges_1, EdgeCounts, numV, LapM, out_dir)
        p = p.cpu().numpy()
        # # write_obj(save_path, p, normals=[], faces=hfaces, vts=[])

        np.save(save_path, p)
        
        print("{} >> {}".format(get_timestamp(), save_path))
        sys.stdout.flush()
    print("#### Done")
    sys.stdout.flush()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch implementation of the paper: Variational auto-encoder for collagen fiber centerline generation and extraction in fibrotic cancer tissues.')
    parser.add_argument("--start-idx", type=int, help="starting index", default=0)
    parser.add_argument("--end-idx", type=int, help="end index", default=-1)
    parser.add_argument("--cuda-idx", type=int, help="cuda index", default=0)
    parser.add_argument("--folder", type=str, help="anger, fear", default="anger")
    parser.add_argument("--out-dir", type=str, help='output directory', default="output/test")
    args = parser.parse_args()
    
    run(args)