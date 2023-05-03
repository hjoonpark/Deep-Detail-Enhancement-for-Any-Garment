from DataIO import *
import torch
import numpy as np
from torch import optim
import os, math
import torch.nn as nn
from Losses import Loss_NormalCrossVert, LapSmooth_loss
import time
import cv2, glob

#USE_CUDA = False
USE_CUDA = torch.cuda.is_available()
print('balin-->', USE_CUDA)
device = torch.device("cuda:0" if USE_CUDA else "cpu")

num_iter = 10000

cEdgeWeight = 1
distWeight = 0
smoothWeight = 0

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
    output = (output*2)-1
    return output

def geo_opt_ours(nmap_path, rrVertArray, uvs, vertEdges_0, vertEdges_1, EdgeCounts, numV, LapM):
    texture_map = np.array(cv2.cvtColor(cv2.imread(nmap_path), cv2.COLOR_BGR2RGB))
    rrNormArray = texel_interpolation(texture_map, uvs)
    rrNormTensor = torch.from_numpy(rrNormArray).type(torch.FloatTensor).to(device)
    print("rrNormArray:", rrNormArray.shape)
    print("rrNormTensor:", rrNormTensor.shape)
    gtVertTensor = torch.from_numpy(rrVertArray).type(torch.FloatTensor).to(device)
    print("gtVertTensor:", gtVertTensor.shape)
    
    # ini vert position
    noise = torch.from_numpy(rrVertArray.copy()).type(torch.FloatTensor).to(device)
    noise.requires_grad = True

    Func_lossNormalCrossVert = Loss_NormalCrossVert(vertEdges_0, vertEdges_1, EdgeCounts, numV, device).to(device)
    Func_lossVertToGtVert = nn.L1Loss(reduction='mean').to(device)

    adam = optim.Adam(params=[noise], lr=0.001, betas=(0.9, 0.999), amsgrad=True)
    oldLoss = 0.
    t = time.time()
    
    for iteration in range(num_iter + 1):
        adam.zero_grad()
        loss_geo = Func_lossNormalCrossVert(normalArray=rrNormTensor, vertArray=noise)
        loss_dist = Func_lossVertToGtVert(gtVertTensor, noise)

        total_loss = cEdgeWeight * loss_geo + distWeight * loss_dist
        if LapM is not None:
            loss_smooth = LapSmooth_loss(LapM, noise)
            total_loss += smoothWeight * loss_smooth
        else:
            loss_smooth = 0

        if iteration % 10 == 0:
            print("Iteration: {}, old_loss: {:.6f}, total Loss: {:.6f}, Geo Loss: {:.6f}, disLoss: {:.6f}, Smooth Loss: {:.6f}".format(iteration,oldLoss, total_loss.item(), cEdgeWeight * loss_geo.item(), distWeight * loss_dist, smoothWeight * loss_smooth))

        # if abs(oldLoss-total_loss.item()) < 0.000001 and iteration > 50:
        if loss_geo < 1e-4:
            print("total_loss.item()-oldLoss:", total_loss.item()-oldLoss)
            return noise.clone().detach()

        oldLoss = total_loss.item()

        # backprop
        total_loss.backward()
        # update parameters
        adam.step()

    return noise.clone().detach()

def geo_opt(gtVertName, rrVertName, rrNormName, ccFlagName, vertEdges_0, vertEdges_1, EdgeCounts, numV, LapM, faceArray):
    gtVertArray = np.array(readVertArrayFile(gtVertName))
    rrVertArray = np.array(readVertArrayFile(rrVertName))
    rrNormArray = np.array(readVertArrayFile(rrNormName))
    rrColorArray = norm_color(rrNormArray)
    ccInds = readVertCCFlag(ccFlagName)
    print(len(ccInds))

    gtVertTensor = torch.from_numpy(gtVertArray).type(torch.FloatTensor).to(device)
    rrNormTensor = torch.from_numpy(rrNormArray).type(torch.FloatTensor).to(device)
    #print(gtVertTensor.size())

    #lapNorm_Corr_activations = LaplacianCorrNorm(LapM=LapM, vert=gtVertTensor, normal=gtNormTensor)

    # ini vert position
    noise = torch.from_numpy(rrVertArray.copy()).type(torch.FloatTensor).to(device)
    noise.requires_grad = True

    Func_lossNormalCrossVert = Loss_NormalCrossVert(vertEdges_0, vertEdges_1, EdgeCounts, numV, device).to(device)
    Func_lossVertToGtVert = nn.L1Loss(reduction='mean').to(device)

    adam = optim.Adam(params=[noise], lr=0.001, betas=(0.9, 0.999), amsgrad=True)
    oldLoss = 0.
    t = time.time()
    for iteration in range(num_iter + 1):
        adam.zero_grad()
        loss_geo = Func_lossNormalCrossVert(normalArray=rrNormTensor, vertArray=noise)
        loss_dist = Func_lossVertToGtVert(gtVertTensor[ccInds, :], noise[ccInds, :])
        loss_smooth = LapSmooth_loss(LapM, noise)
        # betaCorr = 0.8
        # noise_lapNorm_Corr_activations = LaplacianCorrNorm(LapM=LapM, vert=noise, normal=rrNormTensor)
        # loss_smooth = betaCorr * LapNorm_CorrLoss(lapNorm_Corr_activations, noise_lapNorm_Corr_activations, numV) \
        #               + (1.-betaCorr) * LapSmooth_loss(LapM, noise)

        total_loss = cEdgeWeight * loss_geo + smoothWeight * loss_smooth + distWeight * loss_dist
        #total_loss = cEdgeWeight * loss_geo + smoothWeight * loss_smooth
        if iteration % 10 == 0:
            print("Iteration: {}, total Loss: {:.3f}, Geo Loss: {:.3f}, disLoss: {:.3f}, Smooth Loss: {:.3f}".
                  format(iteration, total_loss.item(), cEdgeWeight * loss_geo.item(), distWeight * loss_dist,
                         smoothWeight * loss_smooth.item()))

        # if not os.path.exists('../test/generated_geo/'):
        #     os.mkdir('../test/generated_geo/')

        #if iteration % 10 == 0:
        #    rst = noise.clone().detach()
        #    writePlyV_F_N_C(pDir='../test/generated_geo/iter_{}.ply'.format(iteration),
        #                    verts=rst.cpu().numpy(), normals=rrNormArray, colors=rrColorArray, faces=faceArray)

        if total_loss.item()-oldLoss < 0.01 and iteration > 50:
        #if total_loss.item() < 2.5:
            return noise.clone().detach(), rrNormArray, rrColorArray, time.time()-t

        oldLoss = total_loss.item()

        # backprop
        total_loss.backward()
        # update parameters
        adam.step()

    return noise.clone().detach(), rrNormArray, rrColorArray, time.time()-t

if __name__ == '__main__':
    in_dir = "nvidia_data"
    out_dir = "output"
    os.makedirs(out_dir, exist_ok=1)

    folders = ["amazement"]

    hrestshape, _, hfaces = read_obj(os.path.join(in_dir, "restshape_surf_v1.4.obj"))
    print("hrestshape:", hrestshape.shape, hrestshape.min(), hrestshape.max(), hrestshape.dtype)

    uvs = np.loadtxt(os.path.join(in_dir, "restshape_surf_v1.4_sts.txt"))
    uvs[:,1] = 1-uvs[:,1]
    print("uvs:", uvs.shape, uvs.dtype)

    edges = np.loadtxt(os.path.join(in_dir, "edges_src_trg.txt")).astype(int)
    vertEdges_0 = edges[:,0] # p
    vertEdges_1 = edges[:,1] # q
    EdgeCounts = np.loadtxt(os.path.join(in_dir, "edges_counts.txt")).astype(int)
    numV = len(hrestshape)

    print("edges:", edges.shape, edges.dtype)
    print("EdgeCounts:", EdgeCounts.shape, EdgeCounts.dtype)
    LapM = None

    for folder in folders:
        hres_paths = sorted(glob.glob(os.path.join(in_dir, "BakedUVMaps", folder, "hires*.001*")))
        print("  ", folder, ":", len(hres_paths), "frames")

        for hres_path in hres_paths:
            if "16" not in hres_path:
                continue
            bname = os.path.basename(hres_path).split(".png")[0]
            print(">> ", bname)

            p = geo_opt_ours(hres_path, hrestshape, uvs, vertEdges_0, vertEdges_1, EdgeCounts, numV, LapM)
            p = p.cpu().numpy()

            save_path = os.path.join(out_dir, "{}_hires_{}.obj".format(folder, bname))
            write_obj(save_path, p, normals=[], faces=hfaces, vts=[])
            print(p.shape, ">>", save_path)
            # break
        # break
    print("Done")

    # for FrameID in range(frame0, frame1+1):
    #     gtVertName = vertRoot + str(FrameID).zfill(7) + '.txt'
    #     rrVertName = vertRoot + str(FrameID).zfill(7) + '.txt'
    #     ccFlagName = vertRoot + str(FrameID).zfill(7) + '_f.txt'
    #     rrNormName = normRoot + str(FrameID).zfill(7) + '_n.txt'
    #     rst, rrNormArray, rrColorArray, tt = geo_opt(gtVertName, rrVertName, rrNormName, ccFlagName,
    #                                                  vertEdges_0, vertEdges_1, EdgeCounts, numV, LapM, faceArray)
    #     rst = rst.cpu().numpy()
    #     saveName = saveRoot + str(FrameID).zfill(7)
    #     saveOutVerts(rst, saveName + '.obj')
    #     print("Frame: {}, time:{:.4f}s".format(FrameID, tt))
    #     writePlyV_F_N_C(pDir=saveName + '.ply', verts=rst, normals=rrNormArray, colors=rrColorArray, faces=faceArray)
    #     timeCount = timeCount + tt

    # print('Final Time -->', timeCount)

# if __name__ == '__main__':
#     caseName = 'Chamuse_tango/skirt/'
#     prefRoot = '../Data/case_3/'

#     adjName = prefRoot + caseName + '/uv/10_adjGraph.txt'
#     adj, numV = readAdjFile(adjName)
#     LapM = getLaplacianMatrix(adj)
#     LapM = torch.from_numpy(LapM).type(torch.FloatTensor).to(device)

#     EdgeName = prefRoot + caseName + '/uv/10_crossEdge.txt'
#     vertEdges_0, vertEdges_1, EdgeCounts = readCrossEdges(EdgeName, numV)
#     vertEdges_0 = np.array(vertEdges_0)
#     vertEdges_1 = np.array(vertEdges_1)
#     EdgeCounts = np.array(EdgeCounts)

#     FaceName = prefRoot + caseName + '/uv/Face_10.txt'
#     faceArray = np.array(readFaceIndex(FaceName))

#     vertRoot = prefRoot + caseName + '/10_DsUs_C/'
#     normRoot = "../PatchTest/" + caseName + '/normal_ps/'
#     saveRoot = "../PatchTest/" + caseName + '/geo_ps/'
#     frame0 = 160
#     frame1 = 160
#     timeCount = 0.
#     if not os.path.exists(saveRoot):
#         os.makedirs(saveRoot)
#     for FrameID in range(frame0, frame1+1):
#         gtVertName = vertRoot + str(FrameID).zfill(7) + '.txt'
#         rrVertName = vertRoot + str(FrameID).zfill(7) + '.txt'
#         ccFlagName = vertRoot + str(FrameID).zfill(7) + '_f.txt'
#         rrNormName = normRoot + str(FrameID).zfill(7) + '_n.txt'
#         rst, rrNormArray, rrColorArray, tt = geo_opt(gtVertName, rrVertName, rrNormName, ccFlagName,
#                                                      vertEdges_0, vertEdges_1, EdgeCounts, numV, LapM, faceArray)
#         rst = rst.cpu().numpy()
#         saveName = saveRoot + str(FrameID).zfill(7)
#         saveOutVerts(rst, saveName + '.obj')
#         print("Frame: {}, time:{:.4f}s".format(FrameID, tt))
#         writePlyV_F_N_C(pDir=saveName + '.ply', verts=rst, normals=rrNormArray, colors=rrColorArray, faces=faceArray)
#         timeCount = timeCount + tt

#     print('Final Time -->', timeCount)

