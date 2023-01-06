from copyreg import pickle
import enum
import torch 
import numpy as np 
from torch.utils.data import Dataset 
import pickle


class H36MDataset_HRNet(Dataset):
    """Human3.6M dataset including images."""

    def __init__(self, fname, normalize_2d=True, subjects=[1, 5, 6, 7, 8, 9, 11]):
        pickle_off = open(fname, "rb")
        self.data = pickle.load(pickle_off)

        self.add_data = dict()
        self.add_data['root_joint'] = dict()
        self.add_data['poses_2d_pred_norm'] = dict()

        self.add_data['root_joint_gt'] = dict()
        self.add_data['poses_2d_pred_norm_gt'] = dict()

        # select subjects
        selection_array = np.zeros(len(self.data['subjects']), dtype=bool)
        for s in subjects:
            selection_array = np.logical_or(
                selection_array, (np.array(self.data['subjects']) == s))

        self.data['subjects'] = list(
            np.array(self.data['subjects'])[selection_array])
        cams = ['54138969', '55011271', '58860488', '60457274']
        for cam in cams:
            self.data['poses_2d_pred'][cam] = self.data['poses_2d_pred'][cam][selection_array]
            # gt
            self.data['p2d_gt'][cam] = self.data['p2d_gt'][cam][selection_array]
            self.data['p3d_gt'][cam] = self.data['p3d_gt'][cam][selection_array]
            # self.data['images'][cam] = [str(img) for img in np.array(self.data['images'][cam])[selection_array]]

            if normalize_2d:
                self.add_data['root_joint'][cam] = np.transpose(
                    self.data['poses_2d_pred'][cam], (0, 2, 1))[:, :, [0]]
                self.data['poses_2d_pred'][cam] = (np.transpose(self.data['poses_2d_pred'][cam], (0, 2, 1)) -
                                                   self.add_data['root_joint'][cam]).reshape(-1, 32)
                self.add_data['poses_2d_pred_norm'][cam] = np.linalg.norm(self.data['poses_2d_pred'][cam],
                                                                          ord=2, axis=1, keepdims=True)
                self.data['poses_2d_pred'][cam] /= self.add_data['poses_2d_pred_norm'][cam]

                # gt
                self.add_data['root_joint_gt'][cam] = np.transpose(
                    self.data['p2d_gt'][cam], (0, 2, 1))[:, :, [0]]
                self.data['p2d_gt'][cam] = (np.transpose(self.data['p2d_gt'][cam], (0, 2, 1)) -
                                            self.add_data['root_joint_gt'][cam]).reshape(-1, 32)
                self.add_data['poses_2d_pred_norm_gt'][cam] = np.linalg.norm(self.data['p2d_gt'][cam],
                                                                             ord=2, axis=1, keepdims=True)
                self.data['p2d_gt'][cam] /= self.add_data['poses_2d_pred_norm_gt'][cam]

    def __len__(self):
        return self.data['poses_2d_pred']['54138969'].shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = dict()

        cams = ['54138969', '55011271', '58860488', '60457274']

        for c_idx, cam in enumerate(cams):
            p2d = torch.Tensor(
                self.data['poses_2d_pred'][cam][idx].astype('float32')).cuda()
            sample['cam' + str(c_idx)] = p2d

            # gt ------------------------------------------------------
            p2d_gt = torch.Tensor(
                self.data['p2d_gt'][cam][idx].astype('float32'))
            sample['cam' + str(c_idx) + '_2dgt'] = p2d_gt

            p3d_gt = self.data['p3d_gt'][cam][idx].astype('float32')
            sample['cam' + str(c_idx)+'_3dgt'] = p3d_gt

            # img_set = self.data['images'][cam][idx]
            # sample['cam' + str(c_idx)+'_img'] = str(img_set)

            r2d = self.add_data['root_joint'][cam][idx].astype('float32')
            p2dn = self.add_data['poses_2d_pred_norm'][cam][idx].astype(
                'float32')
            sample['cam' + str(c_idx) + '_joint'] = r2d
            sample['cam' + str(c_idx) + '_norm'] = p2dn

            r2d_gt = self.add_data['root_joint_gt'][cam][idx].astype('float32')
            p2dn_gt = self.add_data['poses_2d_pred_norm_gt'][cam][idx].astype(
                'float32')
            sample['cam' + str(c_idx) + '_joint_gt'] = r2d_gt
            sample['cam' + str(c_idx) + '_norm_gt'] = p2dn_gt


        sample['subjects'] = self.data['subjects'][idx]

        return sample


class H36MDataset(Dataset):
    """Human3.6M dataset including images."""

    def __init__(self, fname, normalize_2d=True, subjects=[1,5,6,7,8,9,11]):
        pickle_off = open(fname, "rb")
        self.data = pickle.load(pickle_off)

        self.add_data = dict()
        self.add_data['root_joint'] = dict() 
        self.add_data['poses_2d_pred_norm'] = dict() 

        self.add_data['root_joint_gt'] = dict() 
        self.add_data['poses_2d_pred_norm_gt'] = dict() 

        # select subjects 
        selection_array = np.zeros(len(self.data['subjects']), dtype=bool)
        for s in subjects:
            selection_array = np.logical_or(selection_array, (np.array(self.data['subjects']) == s))

        self.data['subjects'] = list(np.array(self.data['subjects'])[selection_array])
        cams = ['54138969', '55011271', '58860488', '60457274']
        for cam in cams:
            self.data['poses_2d_pred'][cam] = self.data['poses_2d_pred'][cam][selection_array]
            self.data['confidences'][cam] = self.data['confidences'][cam][selection_array]
            # gt 
            self.data['p2d_gt'][cam] = self.data['p2d_gt'][cam][selection_array]
            self.data['p3d_gt'][cam] = self.data['p3d_gt'][cam][selection_array]
            self.data['images'][cam] = [str(img) for img in np.array(self.data['images'][cam])[selection_array]]

            if normalize_2d:
                self.add_data['root_joint'][cam] = np.transpose(self.data['poses_2d_pred'][cam],(0,2,1))[:,:,[0]]
                self.data['poses_2d_pred'][cam] = (np.transpose(self.data['poses_2d_pred'][cam],(0,2,1)) - \
                                                    self.add_data['root_joint'][cam]).reshape(-1,32)
                self.add_data['poses_2d_pred_norm'][cam] = np.linalg.norm(self.data['poses_2d_pred'][cam],
                                                                            ord=2, axis=1, keepdims=True)
                self.data['poses_2d_pred'][cam] /= self.add_data['poses_2d_pred_norm'][cam]

                # gt
                self.add_data['root_joint_gt'][cam] = np.transpose(self.data['p2d_gt'][cam],(0,2,1))[:,:,[0]]
                self.data['p2d_gt'][cam] = (np.transpose(self.data['p2d_gt'][cam],(0,2,1)) - \
                                                    self.add_data['root_joint_gt'][cam]).reshape(-1,32)
                self.add_data['poses_2d_pred_norm_gt'][cam] = np.linalg.norm(self.data['p2d_gt'][cam],
                                                                            ord=2, axis=1, keepdims=True)
                self.data['p2d_gt'][cam] /= self.add_data['poses_2d_pred_norm_gt'][cam]

    def __len__(self):
        return self.data['poses_2d_pred']['54138969'].shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = dict()

        cams = ['54138969', '55011271', '58860488', '60457274']

        for c_idx, cam in enumerate(cams):
            p2d = torch.Tensor(self.data['poses_2d_pred'][cam][idx].astype('float32')).cuda()
            sample['cam' + str(c_idx)] = p2d 

            #gt ------------------------------------------------------
            p2d_gt = torch.Tensor(self.data['p2d_gt'][cam][idx].astype('float32'))
            sample['cam' + str(c_idx) + '_2dgt'] = p2d_gt 
            
            p3d_gt = self.data['p3d_gt'][cam][idx].astype('float32')
            sample['cam' + str(c_idx)+'_3dgt'] = p3d_gt
            
            img_set = self.data['images'][cam][idx]
            sample['cam' + str(c_idx)+'_img'] = str(img_set)
                      
            r2d = self.add_data['root_joint'][cam][idx].astype('float32')
            p2dn = self.add_data['poses_2d_pred_norm'][cam][idx].astype('float32')
            sample['cam' + str(c_idx) + '_joint'] = r2d
            sample['cam' + str(c_idx) + '_norm'] = p2dn 
            
            r2d_gt = self.add_data['root_joint_gt'][cam][idx].astype('float32')
            p2dn_gt = self.add_data['poses_2d_pred_norm_gt'][cam][idx].astype('float32')  
            sample['cam' + str(c_idx) + '_joint_gt'] = r2d_gt
            sample['cam' + str(c_idx) + '_norm_gt'] = p2dn_gt

        sample['confidences'] = dict()
        for cam in cams:
            sample['confidences'][cam] = torch.Tensor(self.data['confidences'][cam][idx].astype('float32')).cuda()

        sample['subjects'] = self.data['subjects'][idx]

        return sample

class SkiDataset(Dataset):
    """SkiPose-PTZ dataset including images."""

    def __init__(self, fname, normalize_2d=True, subjects=[103, 110, 115, 124, 202, 207, 214, 221, 302, 309, 405, 412]):
        pickle_off = open(fname, "rb")
        self.data = pickle.load(pickle_off)
        
        self.add_data = dict()
        self.add_data['root_joint'] = dict()
        self.add_data['poses_2d_pred_norm'] = dict()        
        # gt
        self.add_data['root_joint_gt'] = dict()
        self.add_data['poses_2d_pred_norm_gt'] = dict()    

        # select subjects
        selection_array = np.zeros(len(self.data['subjects']), dtype=bool)
        for s in subjects:
            selection_array = np.logical_or(selection_array, (np.array(self.data['subjects']) == s))

        self.data['subjects'] = list(np.array(self.data['subjects'])[selection_array])
        cams = ['cam_00', 'cam_01', 'cam_02', 'cam_03','cam_04','cam_05']
        for cam in cams:
            self.data['poses_2d_pred'][cam] = self.data['poses_2d_pred'][cam][selection_array]
            self.data['confidences'][cam] = self.data['confidences'][cam][selection_array]
            # 추가
            self.data['poses_2d_annot'][cam] = self.data['poses_2d_annot'][cam][selection_array]
            self.data['poses_3d_annot'][cam] = self.data['poses_3d_annot'][cam][selection_array]
            self.data['image_path'][cam] = np.array(self.data['image_path'][cam])[selection_array]

            if normalize_2d:

                self.add_data['root_joint'][cam] = np.transpose(self.data['poses_2d_pred'][cam].reshape(-1,16,2),(0,2,1))[:, :, [0]]               # (b,16,2) --> (b,2,16) --> (136399, 2, 1)
                self.data['poses_2d_pred'][cam] = (np.transpose(self.data['poses_2d_pred'][cam].reshape(-1,16,2),(0,2,1)) - self.add_data['root_joint'][cam]).reshape(-1, 32)
                self.add_data['poses_2d_pred_norm'][cam] = np.linalg.norm(self.data['poses_2d_pred'][cam], ord=2, axis=1, keepdims=True)
                self.data['poses_2d_pred'][cam] /= self.add_data['poses_2d_pred_norm'][cam]
                # 추가
                self.add_data['root_joint_gt'][cam] = np.transpose(self.data['poses_2d_annot'][cam].reshape(-1,16,2),(0,2,1))[:, :, [0]]
                self.data['poses_2d_annot'][cam] = (np.transpose(self.data['poses_2d_annot'][cam].reshape(-1,16,2),(0,2,1)) - self.add_data['root_joint_gt'][cam]).reshape(-1, 32)
                self.add_data['poses_2d_pred_norm_gt'][cam] = np.linalg.norm(self.data['poses_2d_annot'][cam], ord=2, axis=1, keepdims=True)
                self.data['poses_2d_annot'][cam] /= self.add_data['poses_2d_pred_norm_gt'][cam]

    def __len__(self):
        return self.data['poses_2d_annot']['cam_00'].shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = dict()

        cams = ['cam_00', 'cam_01', 'cam_02', 'cam_03','cam_04','cam_05']

        for c_idx, cam in enumerate(cams):
            p2d = torch.Tensor(self.data['poses_2d_pred'][cam][idx].astype('float32')).cuda()
            sample['cam' + str(c_idx)] = p2d

            # gt -----------------------------------------------------
            p2d_gt = torch.Tensor(self.data['poses_2d_annot'][cam][idx].astype('float32')).cuda()   # cuda 있을 필요있나?
            sample['cam' + str(c_idx) + '_2dgt'] = p2d_gt 
            
            p3d_gt = self.data['poses_3d_annot'][cam][idx].astype('float32')
            sample['cam' + str(c_idx)+'_3dgt'] = p3d_gt
            
            img_set = self.data['image_path'][cam][idx]
            sample['cam' + str(c_idx)+'_img'] = str(img_set)
                 
            r2d = self.add_data['root_joint'][cam][idx].astype('float32')
            p2dn = self.add_data['poses_2d_pred_norm'][cam][idx].astype('float32')
            sample['cam' + str(c_idx) + '_joint'] = r2d
            sample['cam' + str(c_idx) + '_norm'] = p2dn 
            
            r2d_gt = self.add_data['root_joint_gt'][cam][idx].astype('float32')
            p2dn_gt = self.add_data['poses_2d_pred_norm_gt'][cam][idx].astype('float32')  
            sample['cam' + str(c_idx) + '_joint_gt'] = r2d_gt
            sample['cam' + str(c_idx) + '_norm_gt'] = p2dn_gt

        sample['confidences'] = dict()
        for cam in cams:
            sample['confidences'][cam] = torch.Tensor(self.data['confidences'][cam][idx].astype('float32')).cuda()

        sample['subjects'] = self.data['subjects'][idx]

        return sample

# (161362, 16)

class Dataset3DPW(Dataset):
    """Human3.6M dataset including images."""

    def __init__(self, fname, normalize_2d=True):
        pickle_off = open(fname, "rb")
        self.data = pickle.load(pickle_off)

        self.add_data = dict()
        self.add_data['root_joint'] = dict() 
        self.add_data['p2d_pred_norm'] = dict() 

        # self.add_data['root_joint_gt'] = dict() 
        # self.add_data['p2d_pred_norm_gt'] = dict() 

        self.data['confidences']['cam0'] = self.data['confidences']['cam0'].reshape(-1,16)

        # select subjects 
        cams = ['cam0']
        for cam in cams:
        
            if normalize_2d:
                self.add_data['root_joint'][cam] = np.transpose(self.data['p2d_pred'][cam],(0,2,1))[:,:,[0]]
                self.data['p2d_pred'][cam] = (np.transpose(self.data['p2d_pred'][cam],(0,2,1)) - \
                                                    self.add_data['root_joint'][cam]).reshape(-1,32)
                self.add_data['p2d_pred_norm'][cam] = np.linalg.norm(self.data['p2d_pred'][cam],
                                                                            ord=2, axis=1, keepdims=True)
                self.data['p2d_pred'][cam] /= self.add_data['p2d_pred_norm'][cam]

                # # gt
                # self.add_data['root_joint_gt'][cam] = np.transpose(self.data['p2d_gt'][cam],(0,2,1))[:,:,[0]]
                # self.data['p2d_gt'][cam] = (np.transpose(self.data['p2d_gt'][cam],(0,2,1)) - \
                #                                     self.add_data['root_joint_gt'][cam]).reshape(-1,32)
                # self.add_data['p2d_pred_norm_gt'][cam] = np.linalg.norm(self.data['p2d_gt'][cam],
                #                                                             ord=2, axis=1, keepdims=True)
                # self.data['p2d_gt'][cam] /= self.add_data['p2d_pred_norm_gt'][cam]

    def __len__(self):
        return self.data['p2d_pred']['cam0'].shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = dict()

        cams = ['cam0']

        for c_idx, cam in enumerate(cams):
            p2d = torch.Tensor(self.data['p2d_pred'][cam][idx].astype('float32')).cuda()
            sample['cam' + str(c_idx)] = p2d 

            #gt ------------------------------------------------------
            p2d_gt = torch.Tensor(self.data['p2d_gt'][cam][idx].astype('float32'))
            sample['cam' + str(c_idx) + '_2dgt'] = p2d_gt 
            
            p3d_gt = self.data['p3d_gt'][cam][idx].astype('float32')
            sample['cam' + str(c_idx)+'_3dgt'] = p3d_gt
            
            img_set = self.data['images'][cam][idx]
            sample['cam' + str(c_idx)+'_img'] = str(img_set)
                      
            r2d = self.add_data['root_joint'][cam][idx].astype('float32')
            p2dn = self.add_data['p2d_pred_norm'][cam][idx].astype('float32')
            sample['cam' + str(c_idx) + '_joint'] = r2d
            sample['cam' + str(c_idx) + '_norm'] = p2dn 
            
            # r2d_gt = self.add_data['root_joint_gt'][cam][idx].astype('float32')
            # p2dn_gt = self.add_data['p2d_pred_norm_gt'][cam][idx].astype('float32')  
            # sample['cam' + str(c_idx) + '_joint_gt'] = r2d_gt
            # sample['cam' + str(c_idx) + '_norm_gt'] = p2dn_gt

        sample['confidences'] = dict()
        for cam in cams:
            sample['confidences'][cam] = torch.Tensor(self.data['confidences'][cam][idx].astype('float32')).cuda()


        return sample


