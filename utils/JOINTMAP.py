# root0, RHip1, RKnee2, RAnkle3, LHip4, Lknee5, KAnkle6, Neck7,Nose8,Head9,Lshoulder10,LElbow11,Lwist12,Rshoulder13,RElbow14,Rwist15
H36M_JOINTMAP = [
    [0,1], [1,2], [2,3],      # 오른쪽 하체
    [0,4], [4,5], [5,6],      # 왼쪽 하체 
    [0,7], [7,8], [8,9],      # 중앙 
    [7,10], [10,11], [11,12],    # 왼쪽 상체
    [7,13], [13,14], [14,15]     # 오른쪽 상체
    ]

MPII_JOINTMAP = [
    [0,1],      
    [1,2],      #
    [3,4],      #
    [4,5],      
    [6,0],
    [6,3],
    [6,7],
    [7,8],
    [8,9],
    [7,10],
    [7,13],
    [10,11],
    [11,12],
    [13,14],
    [14,15]
    ]

# joint indice description
joint_names_h36m = ['hip','right_up_leg','right_leg','right_foot','left_up_leg','left_leg', 'left_foot','spine1','neck', 'head','head-top','left-arm','left_forearm','left_hand','right_arm','right_forearm','right_hand']
SKI_JOINTMAP = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], [9, 10], [8, 14], [14, 15], [15, 16], [8, 11], [11, 12], [12, 13]]

