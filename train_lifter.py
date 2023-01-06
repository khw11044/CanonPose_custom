# 학습 안됨


import torch
import torch.nn
import torch.optim
import numpy as np
from torch.utils import data
from utils.data import H36MDataset
import torch.optim as optim
import networks.model_confidences as model_confidences
from utils.print_losses import print_losses
from types import SimpleNamespace
from pytorch3d.transforms import so3_exponential_map as rodrigues
from numpy.random import default_rng
from tqdm import tqdm
from utils.functions import *
import datetime

#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

config = SimpleNamespace()

config.learning_rate = 0.0001
config.BATCH_SIZE = 512
config.N_epochs = 100

# weights for the different losses
config.weight_rep = 1
config.weight_view = 1
config.weight_camera = 0.1

data_folder = './data/'

config.datafile = data_folder + 'alphapose_h36m.pickle'

config.save_model_name =  'models/model_lifter.pt'
config.checkpoint = 'models/chekcpoints'

def loss_weighted_rep_no_scale(p2d, p3d, confs):
    # the weighted reprojection loss as defined in Equation 5

    # normalize by scale
    scale_p2d = torch.sqrt(p2d[:, 0:32].square().sum(axis=1, keepdim=True) / 32)
    p2d_scaled = p2d[:, 0:32]/scale_p2d

    # only the u,v coordinates are used and depth is ignored
    # this is a simple weak perspective projection
    scale_p3d = torch.sqrt(p3d[:, 0:32].square().sum(axis=1, keepdim=True) / 32)
    p3d_scaled = p3d[:, 0:32]/scale_p3d

    loss = ((p2d_scaled - p3d_scaled).abs().reshape(-1, 2, 16).sum(axis=1) * confs).sum() / (p2d_scaled.shape[0] * p2d_scaled.shape[1])

    return loss


def train(config, len_dataset, train_loader, model_skel_morph, model, optimizer, epoch, losses,losses_mean):
    return step('train', config, len_dataset, train_loader, model_skel_morph, model, optimizer, epoch, losses,losses_mean)


def test(config, len_dataset, test_loader, model_skel_morph, model):
    with torch.no_grad():
        return step('test', config, len_dataset, test_loader, model_skel_morph, model)


def step(split, config, len_dataset, dataLoader, model_skel_morph, model, optimizer=None, epoch=None, losses=None,losses_mean=None):
    if split == 'train':
        model.train()
    else:
        model.eval()
        mpjpes_list = []
        pmpjpes_list = []
        pcks_list = []

    for iter, sample in enumerate(tqdm(dataLoader, 0)):

        # not the most elegant way to extract the dictionary
        poses_2d = {key:sample[key] for key in all_cams}
        poses_2d_gt = {key:sample[key+'_2dgt'] for key in all_cams}
        poses_3d_gt = {key:sample[key+'_3dgt'] for key in all_cams} 

        inp_poses = torch.zeros((poses_2d['cam0'].shape[0] * len(all_cams), 32)).cuda()
        inp_confidences = torch.zeros((poses_2d['cam0'].shape[0] * len(all_cams), 16)).cuda()
        gt_confidences = torch.ones((poses_2d['cam0'].shape[0] * len(all_cams), 16)).cuda()

        poses_2dgt = torch.zeros((poses_2d_gt['cam0'].shape[0] * len(all_cams), 32)).cuda()
        poses_3dgt = torch.zeros((poses_3d_gt['cam0'].shape[0] * len(all_cams), 16,3)).cuda()

        # poses_2d is a dictionary. It needs to be reshaped to be propagated through the model.
        cnt = 0
        for b in range(poses_2d['cam0'].shape[0]):
            for c_idx, cam in enumerate(poses_2d):
                inp_poses[cnt] = poses_2d[cam][b]
                inp_confidences[cnt] = sample['confidences'][cam_names[c_idx]][b]
                poses_2dgt[cnt] = poses_2d_gt[cam][b]
                poses_3dgt[cnt] = poses_3d_gt[cam][b]
                    
                cnt += 1

        # morph the poses using the skeleton morphing network
        # inp_poses = model_skel_morph(inp_poses)

        # predict 3d poses
        pred = model(poses_2dgt, gt_confidences)
        pred_poses = pred[0]
        pred_cam_angles = pred[1]

        # angles are in axis angle notation
        # use Rodrigues formula (Equations 3 and 4) to get the rotation matrix
        pred_rot = rodrigues(pred_cam_angles)

        # reproject to original cameras after applying rotation to the canonical poses
        rot_poses = pred_rot.matmul(pred_poses.reshape(-1, 3, 16)).reshape(-1, 48)

# ----------------------------------------------------------------------------------------

        if split == 'test':
            pred_3dpose = rot_poses.reshape(-1, 3, 16).cpu().detach().numpy()
            pred_3dpose = np.transpose(pred_3dpose,(0,2,1)) # (4,16,3)

            poses_3dgt = poses_3dgt.cpu().detach().numpy()
            poses_3dgt, pose_norm, _ = regular_normalized3d(poses_3dgt) 
            pred_3dpose, _, _ = regular_normalized3d(pred_3dpose) 

            poses_3dgt = poses_3dgt * pose_norm
            pred_3dpose = pred_3dpose * pose_norm
            

            mpjpe = np.mean( np.sqrt( np.sum( (poses_3dgt - pred_3dpose)**2, axis=2) ) )
            
            diff = np.sqrt(np.square(poses_3dgt - pred_3dpose).sum(axis=2))
            pck = 100 * len(np.where(diff < 150)[0]) / (diff.shape[0] * diff.shape[1])

            pred_3dpose = pred_3dpose.reshape(-1,4,16,3)
            poses_3dgt = poses_3dgt.reshape(-1,4,16,3)

            pmpjpes = []
            for k in range(len(poses_3dgt)):
                pmpjpe = []
                for v in range(len(all_cams)):
                    mpjpes = np.mean(np.sqrt(np.sum((pred_3dpose[k][v] - poses_3dgt[k][v])**2, axis=1)))
                    pmpjpe.append(mpjpes)
                pmpjpe = min(pmpjpe)
                pmpjpes.append(pmpjpe)
            pmpjpe = np.mean(pmpjpes)

            # print('mpjpe,  pmpjpe,  pck')
            # print('{:.2f} / {:.2f} / {:.2f}'.format( mpjpe, pmpjpe, pck ))

            mpjpes_list.append(mpjpe)
            pmpjpes_list.append(pmpjpe)
            pcks_list.append(pck)

        elif split == 'train':

            # reprojection loss
            losses.rep = loss_weighted_rep_no_scale(inp_poses, rot_poses, inp_confidences)

            # view-consistency and camera-consistency
            # to compute the different losses we need to do some reshaping
            pred_poses_rs = pred_poses.reshape((-1, len(all_cams), 48))
            pred_rot_rs = pred_rot.reshape(-1, len(all_cams), 3, 3)
            confidences_rs = inp_confidences.reshape(-1, len(all_cams), 16)
            inp_poses_rs = inp_poses.reshape(-1, len(all_cams), 32)
            rot_poses_rs = rot_poses.reshape(-1, len(all_cams), 48)

            # view and camera consistency are computed in the same loop
            losses.view = 0
            losses.camera = 0
            for c_cnt in range(len(all_cams)):
                ## view consistency
                # get all cameras and active cameras
                ac = np.array(range(len(all_cams)))
                coi = np.delete(ac, c_cnt)

                # view consistency
                projected_to_other_cameras = pred_rot_rs[:, coi].matmul(pred_poses_rs.reshape(-1, len(all_cams), 3, 16)[:, c_cnt:c_cnt+1].repeat(1, len(all_cams)-1, 1, 1)).reshape(-1, len(all_cams)-1, 48)
                losses.view += loss_weighted_rep_no_scale(inp_poses.reshape(-1, len(all_cams), 32)[:, coi].reshape(-1, 32),
                                                        projected_to_other_cameras.reshape(-1, 48),
                                                        inp_confidences.reshape(-1, len(all_cams), 16)[:, coi].reshape(-1, 16))

                ## camera consistency
                relative_rotations = pred_rot_rs[:, coi].matmul(pred_rot_rs[:, [c_cnt]].permute(0, 1, 3, 2))

                # only shuffle in between subjects
                rng = default_rng()
                for subject in sample['subjects'].unique():
                    # only shuffle if enough subjects are available
                    if (sample['subjects'] == subject).sum() > 1:
                        shuffle_subjects = (sample['subjects'] == subject)
                        num_shuffle_subjects = shuffle_subjects.sum()
                        rand_perm = rng.choice(num_shuffle_subjects.cpu().numpy(), size=num_shuffle_subjects.cpu().numpy(), replace=False)
                        samp_relative_rotations = relative_rotations[shuffle_subjects]
                        samp_rot_poses_rs = rot_poses_rs[shuffle_subjects]
                        samp_inp_poses = inp_poses_rs[shuffle_subjects][:, coi].reshape(-1, 32)
                        samp_inp_confidences = confidences_rs[shuffle_subjects][:, coi].reshape(-1, 16)

                        random_shuffled_relative_projections = samp_relative_rotations[rand_perm].matmul(samp_rot_poses_rs.reshape(-1, len(all_cams), 3, 16)[:, c_cnt:c_cnt+1].repeat(1, len(all_cams)-1, 1, 1)).reshape(-1, len(all_cams)-1, 48)

                        losses.camera += loss_weighted_rep_no_scale(samp_inp_poses,
                                                                    random_shuffled_relative_projections.reshape(-1, 48),
                                                                    samp_inp_confidences)

            # get combined loss
            losses.loss = config.weight_rep * losses.rep + \
                        config.weight_view * losses.view + \
                        config.weight_camera * losses.camera

            optimizer.zero_grad()
            losses.loss.backward()

            optimizer.step()

            for key, value in losses.__dict__.items():
                if key not in losses_mean.__dict__.keys():
                    losses_mean.__dict__[key] = []

                losses_mean.__dict__[key].append(value.item())

            # print progress every 100 iterations
            if not iter % 266:
                # print the losses to the console
                # print_losses(epoch, iter, len(my_dataset) / config.BATCH_SIZE, losses_mean.__dict__, print_keys=not(iter % 1000))
                print_losses(config.N_epochs, epoch, iter, len_dataset / config.BATCH_SIZE, losses_mean.__dict__, print_keys=not(iter % 1000))
                

                # this line is important for logging!
                losses_mean = SimpleNamespace()


        # scheduler.step()

    if split == 'test':
        mpjpes_list.append(mpjpe)
        pmpjpes_list.append(pmpjpe)
        pcks_list.append(pck)
        return np.mean(mpjpes_list), np.mean(pmpjpes_list), np.mean(pcks_list)



if __name__ == '__main__':

    print('training start')

    # loading the H36M dataset
    train_dataset = H36MDataset(config.datafile, normalize_2d=True, subjects=[5,6,7,8])
    train_loader = data.DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)

    test_dataset = H36MDataset(config.datafile, normalize_2d=True, subjects=[9,11])
    test_loader = data.DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)

    # load the skeleton morphing model as defined in Section 4.2
    # for another joint detector it needs to be retrained -> train_skeleton_morph.py
    model_skel_morph = torch.load('models/model_skeleton_morph_S1_gh.pt')
    model_skel_morph.eval()

    # loading the lifting network
    model = model_confidences.Lifter().cuda()

    params = list(model.parameters())

    optimizer = optim.Adam(params, lr=config.learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)

    losses = SimpleNamespace()
    losses_mean = SimpleNamespace()

    cam_names = ['54138969', '55011271', '58860488', '60457274']
    all_cams = ['cam0', 'cam1', 'cam2', 'cam3']

    mpjpes = []
    pmpjpes = []
    pcks = []
    nowTimes = []

    for epoch in range(config.N_epochs):

        train(config,len(train_dataset),train_loader,model_skel_morph,model,optimizer,epoch, losses,losses_mean)

        print('test')

        mpjpe, pmpjpe, pck = test(config,len(test_dataset),test_loader,model_skel_morph,model)

        print('mpjpe,  pmpjpe,  pck')
        print('{:.2f} / {:.2f} / {:.2f}'.format( mpjpe, pmpjpe, pck ))

        now = datetime.datetime.now()
        nowTime = now.strftime('%H:%M:%S')   # 12:11:32

        mpjpes.append(mpjpe)
        pmpjpes.append(pmpjpe)
        pcks.append(pck)
        nowTimes.append(nowTime)

        # save the new trained model every epoch
        torch.save(model, config.save_model_name)

        scheduler.step()

    print('mpjpes',mpjpes)

    print()
    print('---'*10)
    print()

    print('pmpjpes',pmpjpes)

    print()
    print('---'*10)
    print()

    print('pcks',pcks)

    print()
    print('---'*10)
    print()

    print('nowTimes',nowTimes)
    

    print(config.save_model_name)
    print('done')