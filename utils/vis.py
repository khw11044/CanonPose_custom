import numpy as np
import cv2
from scipy import ndimage

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils.JOINTMAP import *

skeleton_color_box = [(0,0,255),(255,0,0),(0,150,0)]
red_skels = [0,1,2,12,13,14]
blue_skels = [3,4,5,9,10,11]
black_skels = [6,7,8]

def show3Dposecam2(vis_3d_poses, ax, radius=40, lcolor='red', cam_view=0,data_type='h36m',angle=(10,-60)):            # channels : (17,3)
    if data_type == 'h36m':
        JOINTMAP = H36M_JOINTMAP
        root_joint_numer = 0
    else:
        JOINTMAP = MPII_JOINTMAP
        root_joint_numer = 6
    
    vals = vis_3d_poses[cam_view].T
    for ind, (i,j) in enumerate(JOINTMAP):
        x, z, y = [np.array([vals[i, c], vals[j, c]]) for c in range(3)]    # connections와 짝을 맞춰서 2개씩 
        ax.plot(x, y, -z, lw=2, c=lcolor)

    RADIUS = radius  # space around the subject

    xroot, yroot, zroot = vals[root_joint_numer, 0], vals[root_joint_numer, 1], vals[root_joint_numer, 2]    # root joint를 기준
        
    ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
    ax.set_zlim3d([-RADIUS + zroot, RADIUS + zroot])
    ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])
    ax.view_init(angle[0], angle[1])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

def show3Dposecam(vis_3d_poses, ax, radius=40, lcolor='red', cam_view=0,data_type='h36m',angle=(10,-60)):            # channels : (17,3)
    if data_type == 'h36m':
        JOINTMAP = H36M_JOINTMAP
        root_joint_numer = 0
    else:
        JOINTMAP = MPII_JOINTMAP
        root_joint_numer = 6
    
    vals = vis_3d_poses[cam_view].T
    for ind, (i,j) in enumerate(JOINTMAP):
        x, z, y = [np.array([vals[i, c], vals[j, c]]) for c in range(3)]    # connections와 짝을 맞춰서 2개씩 
        ax.plot(x, y, -z, lw=2, c=lcolor)

    RADIUS = radius  # space around the subject

    xroot, yroot, zroot = vals[root_joint_numer, 0], vals[root_joint_numer, 1], vals[root_joint_numer, 2]    # root joint를 기준
        
    ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
    ax.set_zlim3d([-RADIUS + zroot, RADIUS + zroot])
    ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])
    ax.view_init(angle[0], angle[1])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    # ax.axes.xaxis.set_ticklabels([])
    # ax.axes.yaxis.set_ticklabels([])
    # ax.axes.zaxis.set_ticklabels([])

def draw_skeleton_2Dimg(vis_2d_poses,vis_pred_2d_poses, ax, data_type='h36m', cam_view=0, image_size=(1280,1000)):
    if data_type == 'h36m':
        JOINTMAP = H36M_JOINTMAP
        root_joint_numer = 0
    else:
        JOINTMAP = MPII_JOINTMAP
        root_joint_numer = 6

    image = np.zeros((image_size[0], image_size[1], 3), np.uint8)

    annot2d_keypoints = vis_2d_poses[cam_view].T
    pred2d_keypoints = vis_pred_2d_poses[cam_view].T
    # 빈 이미지(검정 바탕)에 스켈레톤 그리기 
    for j in range(len(JOINTMAP)):
        child = tuple(np.array(annot2d_keypoints[JOINTMAP[j][0]][:2]).astype(int))
        parent = tuple(np.array(annot2d_keypoints[JOINTMAP[j][1]][:2]).astype(int))
        color = (144, 243, 34)

        cv2.circle(image, parent, 8, (255, 255, 255), -1)
        cv2.line(image, child, parent, color, 3) 

        childpred = tuple(np.array(pred2d_keypoints[JOINTMAP[j][0]][:2]).astype(int))
        parentpred = tuple(np.array(pred2d_keypoints[JOINTMAP[j][1]][:2]).astype(int))
        colorpred = (255, 0, 0)

        cv2.circle(image, parentpred, 8, (255, 0, 0), -1)
        cv2.line(image, childpred, parentpred, colorpred, 3) 

    plt.imshow(image)

def draw_skeleton(annot2d_keypoints, img, ax, data_type='h36m', thin=2,color=(0,212,255),skeleton_color_box = [(0,0,255),(255,0,0),(0,150,0)]):
    if data_type == 'h36m':
        JOINTMAP = H36M_JOINTMAP
        root_joint_numer = 0
    else:
        JOINTMAP = MPII_JOINTMAP
        root_joint_numer = 6
    # skeleton_color_box = [(255,255,255),(255,255,255),(255,255,255)]
    # 빈 이미지(검정 바탕)에 스켈레톤 그리기 
    for j in range(len(JOINTMAP)):
        if j in red_skels:
            lcolor = skeleton_color_box[0]
        elif j in blue_skels:
            lcolor = skeleton_color_box[1]
        else:
            lcolor = skeleton_color_box[2]

        child = tuple(np.array(annot2d_keypoints[JOINTMAP[j][0]][:2]).astype(int))
        parent = tuple(np.array(annot2d_keypoints[JOINTMAP[j][1]][:2]).astype(int))

        cv2.line(img, child, parent, lcolor, thin) 
        # cv2.circle(img, parent, thin+1, color, -1)

    for jo in range(len(annot2d_keypoints)):
        joint = tuple(np.array(annot2d_keypoints[jo]).astype(int))
        cv2.circle(img, joint, thin+1, color, -1)

    # cv2.circle(img, tuple(annot2d_keypoints[0].astype(int)), thin+1, color, -1)
    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    return img

def draw_skeleton_gt_conf(inp_2d_keypoints, gt_2d_keypoints, conf, img, ax, data_type='h36m', thin=1,color=(0,0,255),lcolor=(0, 127, 255),th=0.8):
    if data_type == 'h36m':
        JOINTMAP = H36M_JOINTMAP
        root_joint_numer = 0
    else:
        JOINTMAP = MPII_JOINTMAP
        root_joint_numer = 6

    x=np.linspace(0,img.shape[0],img.shape[0]).astype(int)

    # 빈 이미지(검정 바탕)에 스켈레톤 그리기 
    for j in range(len(JOINTMAP)):
        child_gt = tuple(np.array(gt_2d_keypoints[JOINTMAP[j][0]][:2]).astype(int))
        parent_gt = tuple(np.array(gt_2d_keypoints[JOINTMAP[j][1]][:2]).astype(int))

        cv2.circle(img, parent_gt, thin+1, (255,0,0), -1)
        cv2.line(img, child_gt, parent_gt, (255,127,0), thin) 

        child = tuple(np.array(inp_2d_keypoints[JOINTMAP[j][0]][:2]).astype(int))
        parent = tuple(np.array(inp_2d_keypoints[JOINTMAP[j][1]][:2]).astype(int))
        if conf[JOINTMAP[j][1]] >= th:
            cv2.circle(img, parent, thin+1, color, -1)
        else:
            cv2.circle(img, parent, thin+1, (0,255,0), -1)
        cv2.line(img, child, parent, lcolor, thin) 

    cv2.circle(img, tuple(inp_2d_keypoints[0].astype(int)), thin+1, (255,127,127), -1)


    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return img


def show3Dpose_with_annot(annot, vals, ax, radius=40, data_type='h36m', lcolor='red', rcolor='#0000ff',angles=(10,-60)):
    if data_type == 'h36m':
        JOINTMAP = H36M_JOINTMAP
        root_joint_numer = 0
    else:
        JOINTMAP = MPII_JOINTMAP
        root_joint_numer = 6

    for ind, (i,j) in enumerate(JOINTMAP):
        x, z, y = [np.array([annot[i, c], annot[j, c]]) for c in range(3)]
        ax.plot(x, y, -z, lw=2, c='black')

    for ind, (i,j) in enumerate(JOINTMAP):
        if ind in red_skels:
            lcolor = 'red'
        elif ind in blue_skels:
            lcolor = 'blue'
        else:
            lcolor = 'green'
        x, z, y = [np.array([vals[i, c], vals[j, c]]) for c in range(3)]
        ax.plot(x, y, -z, lw=3, c=lcolor)

    # a = 15
    # ax.scatter(annot[a][0], annot[a][2], -annot[a][1], c='red', marker='o', s=15)


    RADIUS = radius  # space around the subject

    xroot, yroot, zroot = vals[root_joint_numer, 0], vals[root_joint_numer, 1], vals[root_joint_numer, 2]

    ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
    ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])
    ax.set_zlim3d([-RADIUS + zroot, RADIUS + zroot])
    ax.view_init(angles[0], angles[-1])
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    # ax.set_zlabel("z")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

def show3Dpose(annot, ax, radius=40, data_type='h36m', lcolor='red', rcolor='#0000ff',angles=(10,-60)):
    if data_type == 'h36m':
        JOINTMAP = H36M_JOINTMAP
        root_joint_numer = 0
    else:
        JOINTMAP = MPII_JOINTMAP
        root_joint_numer = 6

    for ind, (i,j) in enumerate(JOINTMAP):
        x, z, y = [np.array([annot[i, c], annot[j, c]]) for c in range(3)]
        ax.plot(x, y, -z, lw=2, c='black')

    RADIUS = radius  # space around the subject

    xroot, yroot, zroot = annot[root_joint_numer, 0], annot[root_joint_numer, 1], annot[root_joint_numer, 2]

    ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
    ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])
    ax.set_zlim3d([-RADIUS + zroot, RADIUS + zroot])
    ax.view_init(angles[0], angles[-1])
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    # ax.set_zlabel("z")

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])


def show2Dpose(p2d, image, data_type='h36m', colorpred='#dc143c'):
    l_size = 10
    c_size = 15
    if data_type == 'h36m':
        JOINTMAP = H36M_JOINTMAP
        root_joint_numer = 0
    else:
        JOINTMAP = MPII_JOINTMAP
        root_joint_numer = 6

    # image = np.zeros((image_size[0], image_size[1], 3), np.uint8)

    # pred2d_keypoints = p2d[cam_view].T
    # 빈 이미지(검정 바탕)에 스켈레톤 그리기 
    for ind, (i,j) in enumerate(JOINTMAP):
        if ind in red_skels:          # 오른쪽 
            color = (0,0,255)   # 'b'    
        elif ind in blue_skels:       # 왼쪽 
            color = (255,0,0) # 'r'
        else:
            color = (0,255,0) # 'g'        # 중앙

        childpred = tuple(np.array(p2d[i][:2]).astype(int))
        parentpred = tuple(np.array(p2d[j][:2]).astype(int))
        # colorpred = (255, 0, 0)

        cv2.circle(image, parentpred, c_size, (255, 0, 0), -1)
        cv2.line(image, childpred, parentpred, color, l_size) 

    # cv2.circle(image, tuple(np.array(p2d[0][:2]).astype(int)), 8, (0, 0, 0), -1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    return image