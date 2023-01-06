


import torch
from torch import absolute
import torch.nn
import torch.optim
import numpy as np
import matplotlib.pyplot as plt 
from torch.utils import data
import torch.optim as optim
import networks.model_confidences as model_confidences
# from utils.data import H36MDataset
from utils.data import H36MDataset_HRNet as H36MDataset
from utils.functions import *
from utils.vis import *

from types import SimpleNamespace
from pytorch3d.transforms import so3_exponential_map as rodrigues


import sys
from torch.multiprocessing import Pool, Process, set_start_method
try:
     set_start_method('spawn')
except RuntimeError:
    pass


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
GPU_NUM = 0
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')


config = SimpleNamespace()

config.BATCH_SIZE = 1
config.morph_model = 'models/model_skeleton_morph_S1_gh.pt'
config.load_model = 'models/model_lifter.pt'  
# model_lifter
# model_lifter_h36m_fusion


config.vis = False

data_folder = './data/'

config.datafile = data_folder + 'hrnet_dataset.pkl' 
# 'alphapose_2d3dgt_img_h36m.pickle'

if config.vis:
    experiments_image_save = 'experiments/eval_img'
    if not os.path.isdir(experiments_image_save):
        os.mkdir(experiments_image_save)


cam_names = ['54138969', '55011271', '58860488', '60457274']
all_cams = ['cam0', 'cam1', 'cam2', 'cam3']

Radius=1000
# ---------------------------------------------------------------------------------------------------------------------------------------------------
def demo():
    # loading the H36M dataset
    my_dataset = H36MDataset(config.datafile, normalize_2d=True, subjects=[9,11])    # 5,6,7,8 , 9,11
    train_loader = data.DataLoader(my_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)

    model_skel_morph = torch.load(config.morph_model)
    model_skel_morph.eval()


    model = torch.load(config.load_model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    mpjpes = []
    pmpjpes = []
    pcks = []
    if config.vis:
        plt.ion()
        fig = plt.figure(figsize=(16,8))

    for i, sample in enumerate(train_loader):
        if i >= 0:
        # if i % 64 == 0:
            print(i, '/' ,len(train_loader))
            # not the most elegant way to extract the dictionary
            poses_2d = {key:sample[key] for key in all_cams}
            poses_2d_gt = {key:sample[key+'_2dgt'] for key in all_cams}
            poses_3d_gt = {key:sample[key+'_3dgt'] for key in all_cams} 

            joints_2d = {key:sample[key+'_joint'] for key in all_cams} 
            norm_2d = {key:sample[key+'_norm'] for key in all_cams}
            joints_2d_gt = {key:sample[key+'_joint_gt'] for key in all_cams} 
            norm_2d_gt = {key:sample[key+'_norm_gt'] for key in all_cams}


            inp_poses = torch.zeros((poses_2d['cam0'].shape[0] * len(all_cams), 32)).cuda()

            gt_confidences = torch.ones((poses_2d['cam0'].shape[0] * len(all_cams), 16)).cuda()

            poses_2dgt = torch.zeros((poses_2d_gt['cam0'].shape[0] * len(all_cams), 32)).cuda()
            poses_3dgt = torch.zeros((poses_3d_gt['cam0'].shape[0] * len(all_cams), 16,3)).cuda()

            inp_root = torch.zeros((joints_2d['cam0'].shape[0] * len(all_cams), 2,1)).cuda()
            inp_norm_2d = torch.zeros((norm_2d['cam0'].shape[0] * len(all_cams), 1)).cuda()
            gt_root = torch.zeros((joints_2d_gt['cam0'].shape[0] * len(all_cams), 2,1)).cuda()
            gt_norm_2d = torch.zeros((norm_2d_gt['cam0'].shape[0] * len(all_cams), 1)).cuda()


            # poses_2d is a dictionary. It needs to be reshaped to be propagated through the model.
            cnt = 0
            for b in range(poses_2d['cam0'].shape[0]):
                for c_idx, cam in enumerate(poses_2d):
                    inp_poses[cnt] = poses_2d[cam][b]

                    poses_2dgt[cnt] = poses_2d_gt[cam][b]
                    poses_3dgt[cnt] = poses_3d_gt[cam][b]
                    inp_root[cnt] = joints_2d[cam][b]
                    inp_norm_2d[cnt] = norm_2d[cam][b]
                    gt_root[cnt] = joints_2d_gt[cam][b]
                    gt_norm_2d[cnt] = norm_2d_gt[cam][b]
                    
                    cnt += 1

            

# ---------------------------------------------- Lifting -----------------------------------------------

            # morph the poses using the skeleton morphing network
            # inp_poses = model_skel_morph(inp_poses)

            # predict 3d poses
            pred = model(poses_2dgt, gt_confidences)
            pred_poses = pred[0]
            pred_cam_angles = pred[1]

            # angles are in axis angle notation
            # use Rodrigues formula (Equations 3 and 4) to get the rotation matrix
            pred_rot = rodrigues(pred_cam_angles)       # batchsize * 4, 3 --> batchsize * 4, 3, 3

            # reproject to original cameras after applying rotation to the canonical poses
            rot_poses = pred_rot.matmul(pred_poses.reshape(-1, 3, 16)).reshape(-1, 48)  
            # 예측한 3d를 scale한다
            # scaled_pred_3dpose = scaled_normalized3d(rot_poses).reshape(-1, 3, 16).cpu().detach().numpy()
            pred_3dpose = rot_poses.reshape(-1, 3, 16).cpu().detach().numpy()
            pred_3dpose = np.transpose(pred_3dpose,(0,2,1)) # (4,16,3)

            poses_3dgt = poses_3dgt.cpu().detach().numpy()
            poses_3dgt, pose_norm, _ = regular_normalized3d(poses_3dgt) 
            pred_3dpose, _, _ = regular_normalized3d(pred_3dpose) 

            poses_3dgt = poses_3dgt * pose_norm
            pred_3dpose = pred_3dpose * pose_norm
            
            mpjpe = sum([np.mean(np.sqrt(np.sum((pred_3dpose[i]- poses_3dgt[i])**2, axis=1))) for i in range(len(all_cams))]) / len(all_cams)
            mpjpes.append(mpjpe)
            print('mpjpe:',mpjpe)

            pmpjpe = []
            for v in range(len(all_cams)):
                mpjpe = np.mean(np.sqrt(np.sum((pred_3dpose[v] - poses_3dgt[v])**2, axis=1)))
                pmpjpe.append(mpjpe)
            pmpjpe = min(pmpjpe)
            print('pmpjpe:',pmpjpe)
            pmpjpes.append(pmpjpe)

            diff = np.sqrt(np.square(poses_3dgt - pred_3dpose).sum(axis=2))
            pck = 100 * len(np.where(diff < 150)[0]) / (diff.shape[0] * diff.shape[1])
            print('pck:',round(pck))
            pcks.append(pck)

            inp_poses_2D_all = ((inp_poses* gt_norm_2d).reshape(-1,2,16) + gt_root).cpu().detach().numpy() 
            poses_2dgt_2D_all = ((poses_2dgt * gt_norm_2d).reshape(-1,2,16) + gt_root).cpu().detach().numpy() 

                    
            if config.vis:
                for view in range(len(all_cams)):
                    inp_poses_2D = inp_poses_2D_all[view]
                    poses_2dgt_2D = poses_2dgt_2D_all[view]
                    pose_3D = pred_3dpose[view]
                    pose_3D_gt = poses_3dgt[view]

                    ax_3D = fig.add_subplot(3,5,1+view, projection='3d', aspect='auto')
                    show3Dpose_with_annot(pose_3D_gt, pose_3D, ax_3D, data_type='h36m', radius=Radius,angles=(20,-70))
                    # show3Dpose2(pose_3D, ax_3D, data_type='mpii', radius=Radius, lcolor='black',angles=(20,-70))
                    

                print('drawing....')
                plt.draw()
                # plt.savefig(config.save_img_folder + '/%05d.png'% (i),transparent=True)
                #plt.savefig(config.save_img_folder + '/%05d.png'% (i))
                plt.pause(0.1)
                plt.show()
                fig.clear()

    # print(len(mpjpes))
    print('MPJE', np.mean(mpjpes))
    print('PMPJE', np.mean(pmpjpes))
    print('PCK',np.mean(pcks))

if __name__ == '__main__':
    demo()
    print('done')
