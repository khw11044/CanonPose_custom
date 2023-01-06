import torch 
import numpy as np

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


def loss_weighted_ave_pred_3d(ave, p3d, confs):
    # the weighted reprojection loss as defined in Equation 5

    # normalize by scale
    scale_ave = torch.sqrt(ave[:, 0:48].square().sum(axis=1, keepdim=True) / 48)
    ave_scaled = ave[:, 0:48]/scale_ave

    # only the u,v coordinates are used and depth is ignored
    # this is a simple weak perspective projection
    scale_p3d = torch.sqrt(p3d[:, 0:48].square().sum(axis=1, keepdim=True) / 48)
    p3d_scaled = p3d[:, 0:48]/scale_p3d

    loss = ((ave_scaled - p3d_scaled).abs().reshape(-1, 3, 16).sum(axis=1) * confs).sum() / (ave_scaled.shape[0] * ave_scaled.shape[1])

    return loss

def scaled_normalized2d(pose):    # 
    scale_norm = torch.sqrt(pose[:, 0:32].square().sum(axis=1, keepdim=True) / 32)                    # 3d pose도 scaling 필요 
    p3d_scaled = pose[:, 0:32]/scale_norm
    return p3d_scaled.reshape(-1, 2, 16).cpu().detach().numpy(), scale_norm.cpu().detach().numpy()

def scaled_normalized3d(pose):    # 
    scale_norm = torch.sqrt(pose[:, 0:48].square().sum(axis=1, keepdim=True) / 48)
    scaled_pose = pose[:, 0:48]/scale_norm
    return scaled_pose.reshape(-1, 3, 16).cpu().detach().numpy(), scale_norm.cpu().detach().numpy()



def regular_normalized3d(poseset):
    pose_norm_list = []
    pose_root_list = []

    for i in range(len(poseset)):
        root_joints = poseset[i][0] 
        pose_root_list.append(root_joints.copy())
        poseset[i] = (poseset[i] - root_joints)                                     
        pose_norm = np.linalg.norm(poseset[i].T.reshape(-1, 48), ord=2, axis=1, keepdims=True)  
                   
        poseset[i] /= pose_norm
        pose_norm_list.append(pose_norm)
        

    return poseset, np.array(pose_norm_list), np.array(pose_root_list).reshape(-1,1,3)


def get_denormalized_pose(inp_poses,joints_2d,norm_2d):
    vis_input_2d_poses, scale_norm = scaled_normalized2d(inp_poses)

    cnt = 0
    for b in range(joints_2d['cam0'].shape[0]):
        for rj_idx, rj in enumerate(joints_2d):
            vis_input_2d_poses[cnt][0] =  (vis_input_2d_poses[cnt][0] * norm_2d[rj][b].cpu().detach().numpy()[0] * scale_norm[0]) + joints_2d[rj][b].cpu().detach().numpy()[0][0] 
            vis_input_2d_poses[cnt][1] = (vis_input_2d_poses[cnt][1] * norm_2d[rj][b].cpu().detach().numpy()[0] * scale_norm[0]) + joints_2d[rj][b].cpu().detach().numpy()[1][0] 

            cnt += 1
    return np.transpose(vis_input_2d_poses,(0,2,1))


# denormalize pose 
def denormalize_pose(poses,norm_2d,root):
    poses = poses.reshape(-1,2,16).permute(0,2,1)
    denorm_pose = (poses * norm_2d.reshape(-1,1,1)).cpu().detach().numpy()  # + inp_root
    denorm_pose = denorm_pose + root.permute(0,2,1).cpu().detach().numpy()
    return denorm_pose 

def each_joints_jdrs(half_head,mypose,gtpose,joints_jdr_lists):
    joints_distances = np.sqrt(np.sum((mypose - gtpose)**2, axis=1))          # 
    each_joints_jdrs =[1 if jd < half_head else 0 for jd in joints_distances]
    for j,jdr in enumerate(each_joints_jdrs):
        joints_jdr_lists[j].append(jdr)

def reprojection(p2d, p3d):
    # the weighted reprojection loss as defined in Equation 5

    # normalize by scale
    scale_p2d = torch.sqrt(p2d[:, 0:32].square().sum(axis=1, keepdim=True) / 32)
    p2d_scaled = p2d[:, 0:32]/scale_p2d

    # only the u,v coordinates are used and depth is ignored
    # this is a simple weak perspective projection
    scale_p3d = torch.sqrt(p3d[:, 0:32].square().sum(axis=1, keepdim=True) / 32)
    p3d_scaled = p3d[:, 0:32]/scale_p3d
    
    reprojection_2d = p3d_scaled * scale_p2d

    return reprojection_2d


