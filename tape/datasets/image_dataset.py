from os.path import join
import os
from tracemalloc import start
from turtle import Turtle
import cv2
import torch
import numpy as np
from typing import Dict 
from yacs.config import CfgNode
import sys
from .dataset import Dataset
from .utils import get_example, get_example2,rot_aa,split_into_chunks,split_into_chunks2
from lib.models import spin
from lib.data_utils.feature_extractor import extract_features
import joblib
def rectify_poseUP(pose):
    """
    Rectify "upside down" people in global coord
 
    Args:
        pose (72,): Pose.

    Returns:
        Rotated pose.
    """
    pose = pose.copy()
    R_mod = cv2.Rodrigues(np.array([np.pi, 0, 0]))[0]
    R_root = cv2.Rodrigues(pose[:3])[0]
    new_root = R_root.dot(R_mod)
    pose[:3] = cv2.Rodrigues(new_root)[0].reshape(3)
    return pose

def get_feat(start_index, end_index, seqlen=16):
    if start_index != end_index:
        return [i for i in range(start_index, end_index+1)]
    else:
        return [start_index for _ in range(seqlen)]

def get_sequence(self, start_index, end_index, data):
    return data[start_index:end_index+1].copy()
    if start_index != end_index:
        return data[start_index:end_index+1]
    else:
        return data[start_index:start_index+1].repeat(self.seqlen, axis=0)

class ImageDataset(Dataset):
    def __init__(self,
                 cfg: CfgNode,
                 dataset_file: str,
                 img_dir: str,
                 train: bool = False,
                 **kwargs):
        """
        Dataset class used for loading images and corresponding annotations.
        Args:
            cfg (CfgNode): Model config file.
            dataset_file (str): Path to npz file containing dataset info.
            img_dir (str): Path to image folder.
            train (bool): Whether it is for training or not (enables data augmentation).
        """
        super(ImageDataset, self).__init__()
        print(dataset_file)
        self.train = train
        self.cfg = cfg
        self.wimg = cfg.MODE.IMG
        self.img_size = cfg.MODEL.IMAGE_SIZE
        self.mean = 255. * np.array(self.cfg.MODEL.IMAGE_MEAN)
        self.std = 255. * np.array(self.cfg.MODEL.IMAGE_STD)
        self.img=cfg.MODE.IMG
        self.img_dir = img_dir
        try:
            self.data = np.load(dataset_file,allow_pickle=True)
            #self.valid_frames=np.load('/media/nikos/Samsung1TB/egocentric_color/recording_20210907_S03_S04_01/2021-09-07-164904/valid_frame.npz',allow_pickle=True)
        except:
            self.data = joblib.load(dataset_file)
        self.dataset_file = dataset_file
        #try:
        #    self.imgname = self.valid_frames['imgname'][self.valid_frames['valid']] #self.data['imgname']
        #try:
        #print(self.data.keys())
        self.imgname = self.data['imgname']
        #import ipdb
        #ipdb.set_trace()
        self.seqlen = cfg.MODEL.SEQLEN
        self.mid_frame = int(self.seqlen/2)
        self.stride = cfg.MODEL.STRIDE
        path_list=[]
        #if(dataset_file == 'data/datasets/3dpw_train.npz'):
        #    self.stride=3
        #self.stride=5
        for k in self.imgname:
            #if(dataset_file == 'data/datasets/mpi_inf_3dhp_train.npz'):
            #    import ipdb
            #    ipdb.set_trace()

            if(dataset_file == 'data/datasets/mpii_train_spin.npz'):
                path_list.append(k.decode('UTF-8').split(os.sep))
            else:
                path_list.append(k.split(os.sep)[-2])
        arr = np.array(path_list)
        #self.vid_indices =split_into_chunks2(arr, self.seqlen, self.stride, is_train=self.train,match_vibe=Match_vibe)# split_into_chunks(arr,self.seqlen,self.stride)

        self.vid_indices = split_into_chunks(arr,self.seqlen,self.stride)
        #if self.cfg.MODE.TCMR:
        #    self.vid_indices =split_into_chunks2(arr, self.seqlen, self.seqlen, is_train=self.train)# split_into_chunks(arr,self.seqlen,self.stride)
        #self.vid_indices = split_into_chunks2(arr, self.seqlen, self.stride,is_train=train,match_vibe=True)#TODO use parameters for seqlen and stride

        #if self.train:
        #    self.vid_indices = split_into_chunks2(arr, self.seqlen, self.stride,is_train=train,match_vibe=False)#TODO use parameters for seqlen and stride
        #else:
        #    self.vid_indices = split_into_chunks2(arr, self.seqlen, 1,is_train=train,match_vibe=False)#TODO use parameters for seqlen and stride
        body_permutation = [0, 1, 5, 6, 7, 2, 3, 4, 8, 12, 13, 14, 9, 10, 11, 16, 15, 18, 17, 22, 23, 24, 19, 20, 21]
        extra_permutation = [5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13, 14, 15, 16, 17, 18]
        flip_keypoint_permutation = body_permutation + [25 + i for i in extra_permutation]
        self.flip_keypoint_permutation = flip_keypoint_permutation

        num_pose = 3 * (self.cfg.SMPL.NUM_BODY_JOINTS + 1)
        # Bounding boxes are assumed to be in the center and scale format
        self.center = self.data['center']
        self.scale = self.data['scale'].reshape(len(self.center), -1).max(axis=-1)  / 200.0

        # Get gt SMPLX parameters, if available
        try:
            self.body_pose = self.data['body_pose'].astype(np.float32)
            self.has_body_pose = self.data['has_body_pose'].astype(np.float32)
        except KeyError:
            self.body_pose = np.zeros((len(self.imgname), num_pose), dtype=np.float32)
            self.has_body_pose = np.zeros(len(self.imgname), dtype=np.float32)
        try:
            self.betas = self.data['betas'].astype(np.float32)
            self.has_betas = self.data['has_betas'].astype(np.float32)
        except KeyError:
            self.betas = np.zeros((len(self.imgname), 10), dtype=np.float32)
            self.has_betas = np.zeros(len(self.imgname), dtype=np.float32)

        # Try to get 2d keypoints, if available
        try:
            body_keypoints_2d = self.data['body_keypoints_2d']
        except KeyError:
            body_keypoints_2d = np.zeros((len(self.center), 25, 3))
        #EGOBODY
        #body_keypoints_2d=self.data['keypoints']
        # Try to get extra 2d keypoints, if available
        try:
            extra_keypoints_2d = self.data['extra_keypoints_2d']
        except KeyError:
            extra_keypoints_2d = np.zeros((len(self.center), 19, 3))
        self.keypoints_2d = np.concatenate((body_keypoints_2d, extra_keypoints_2d), axis=1).astype(np.float32)
        #self.keypoints_2d = self.data['keypoints']
        #import ipdb
        #ipdb.set_trace()
        # Try to get 3d keypoints, if available
        try:
            body_keypoints_3d = self.data['body_keypoints_3d'].astype(np.float32)
        except KeyError:
            body_keypoints_3d = np.zeros((len(self.center), 25, 4), dtype=np.float32)
        # Try to get extra 3d keypoints, if available
        try:
            extra_keypoints_3d = self.data['extra_keypoints_3d'].astype(np.float32)
        except KeyError:
            extra_keypoints_3d = np.zeros((len(self.center), 19, 4), dtype=np.float32)

        body_keypoints_3d[:, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], -1] = 0

        self.keypoints_3d = np.concatenate((body_keypoints_3d, extra_keypoints_3d), axis=1).astype(np.float32)
        if not self.cfg.MODE.EXTRACT and (not self.cfg.MODE.IMG):
            self.features=self.data['features']
        #import ipdb
        #ipdb.set_trace()

    #def __len__(self) -> int:#TODO maybe uncomment
    #    return len(self.scale)

    def __len__(self):
        return len(self.vid_indices)
    

    def __getitem__(self, idx: int) -> Dict:
        """
        Returns an example from the dataset.
        """
        #if not self.train:
        #    print(self.vid_indices[idx])
        start_index, end_index = self.vid_indices[idx]
        #bbox = [min(self.data['keypoints'][idx][:,0]), min(self.data['keypoints'][idx][:,1]),
        #    max(self.data['keypoints'][idx][:,0]), max(self.data['keypoints'][idx][:,1])]
        #center = [(bbox[2]+bbox[0])/2, (bbox[3]+bbox[1])/2]
        #scale = 1.2*max(bbox[2]-bbox[0], bbox[3]-bbox[1])
        #print(self.center[idx])
        #start_index=0
        #end_index=0
        #print(self.imgname.shape)
        image_file=[]
        for i in range(start_index,end_index+1):
            try:
                image_file.append(join(self.img_dir, self.imgname[i].decode('utf-8')))
            except AttributeError:
                image_file.append(join(self.img_dir, self.imgname[i]))
        if not self.cfg.MODE.EXTRACT and not self.cfg.MODE.IMG:
            features = get_sequence(self, start_index, end_index, self.features)

        #print(image_file)
        keypoints_2d = get_sequence(self, start_index, end_index, self.keypoints_2d) #self.keypoints_2d[start_index:end_index + 1].copy()
        keypoints_3d = get_sequence(self, start_index, end_index, self.keypoints_3d) #self.keypoints_3d[start_index:end_index + 1].copy()
        #print(keypoints_2d.shape)
        center = get_sequence(self, start_index, end_index, self.center)#self.center[start_index:end_index + 1].copy()
        center_x = center[:,1]
        center_y = center[:,0]
 
        bbox_size = get_sequence(self, start_index, end_index, self.scale)*200#self.scale[start_index:end_index + 1]*200
        #import ipdb
        #ipdb.set_trace()
        body_pose = get_sequence(self, start_index, end_index, self.body_pose).astype(np.float32)#self.body_pose[start_index:end_index + 1].copy().astype(np.float32)
        betas = get_sequence(self, start_index, end_index, self.betas).astype(np.float32)#self.betas[start_index:end_index + 1].copy().astype(np.float32)

        has_body_pose = get_sequence(self, start_index, end_index, self.has_body_pose)#self.has_body_pose[start_index:end_index + 1].copy()
        has_betas = get_sequence(self, start_index, end_index, self.has_body_pose)#self.has_betas[start_index:end_index + 1].copy()

        smpl_params_is_axis_angle = {'global_orient': True,
                                     'body_pose': True,
                                     'betas': False
                                    }

        augm_config = self.cfg.DATASETS.CONFIG
        # Crop image and (possibly) perform data augmentation
        #img_patch, keypoints_2d, keypoints_3d, smpl_params, has_smpl_params, img_size
        img_patch=[]
        nkeypoints_2d = []
        nkeypoints_3d=[]
        nsmpl_params = {'global_orient': [],
                       'body_pose': [],
                       'betas': []
                      }
        nhas_smpl_params = {'global_orient': [],
                            'body_pose': [],
                           'betas': []
                        }
        img_size=[]
        if not self.cfg.MODE.EXTRACT and not self.cfg.MODE.IMG:
            feats=[]
        for idx in range(end_index + 1-start_index):
            #idx = idx -1
            if self.train and self.dataset_file == 'data/datasets/h36m_mosh_pav_train_25.npz':
                body_pose[idx]=rectify_poseUP(body_pose[idx])
            smpl_params = {'global_orient': body_pose[idx,:3],
                        'body_pose': body_pose[idx,3:],
                        'betas': betas[idx]
                        }
            has_smpl_params = {'global_orient': has_body_pose[idx],
                        'body_pose': has_body_pose[idx],
                        'betas': has_betas[idx]
                        }
            if(self.train and self.wimg):
                augm=True
            else:
                augm= False
            _img_patch, _keypoints_2d, _keypoints_3d, _smpl_params, _has_smpl_params, _img_size = get_example(image_file[idx],
                                                                                                        center_x[idx], center_y[idx],
                                                                                                        bbox_size[idx], bbox_size[idx],
                                                                                                        keypoints_2d[idx], keypoints_3d[idx],
                                                                                                        smpl_params,
                                                                                                        has_smpl_params,
                                                                                                        self.flip_keypoint_permutation,
                                                                                                        self.img_size, self.img_size,
                                                                                                        self.mean, self.std,augm,
                                                                                                         augm_config,self.img)
            img_patch.append(_img_patch)
            nkeypoints_2d.append(_keypoints_2d)
            nkeypoints_3d.append(_keypoints_3d)
            nsmpl_params['betas'].append(_smpl_params['betas'])
            nsmpl_params['body_pose'].append(_smpl_params['body_pose'])
            nsmpl_params['global_orient'].append(_smpl_params['global_orient'])
            nhas_smpl_params['betas'].append(_has_smpl_params['betas'])
            nhas_smpl_params['body_pose'].append(_has_smpl_params['body_pose'])
            nhas_smpl_params['global_orient'].append(_has_smpl_params['global_orient'])
            img_size.append(_img_size)
        img_patch=np.array(img_patch)
        keypoints_2d = np.array(nkeypoints_2d)
        keypoints_3d=np.array(nkeypoints_3d)
        img_size = np.array(img_size)

        smpl_params['betas']=np.array(nsmpl_params['betas'])
        smpl_params['global_orient']=np.array(nsmpl_params['global_orient'])
        smpl_params['body_pose']=np.array(nsmpl_params['body_pose'])
        has_smpl_params['betas']=np.array(nhas_smpl_params['betas'])
        has_smpl_params['global_orient']=np.array(nhas_smpl_params['global_orient'])
        has_smpl_params['body_pose']=np.array(nhas_smpl_params['body_pose'])
        item = {}

        #import ipdb
        #ipdb.set_trace()
        if(self.dataset_file == 'data/datasets/h36m_mosh_pav_train_25.npz'):
            tmp1=np.zeros(len(nhas_smpl_params['betas']))
            tmp1[::5]=1
            tmp2=np.zeros(len(nhas_smpl_params['global_orient']))
            #tmp2[::5]=1
            tmp3=np.zeros(len(nhas_smpl_params['body_pose']))
            tmp3[::5]=1
            has_smpl_params['betas']=tmp1#np.array(nhas_smpl_params['betas'])
            has_smpl_params['global_orient']=tmp2#np.array(nhas_smpl_params['global_orient'])
            has_smpl_params['body_pose']=tmp3
            #has_smpl_params['betas']=np.array(nhas_smpl_params['betas'])
            has_smpl_params['global_orient']=np.array(nhas_smpl_params['global_orient'])
            has_smpl_params['body_pose']=np.array(nhas_smpl_params['body_pose'])
        else:
                has_smpl_params['betas']=np.array(nhas_smpl_params['betas'])
                has_smpl_params['global_orient']=np.array(nhas_smpl_params['global_orient'])
                has_smpl_params['body_pose']=np.array(nhas_smpl_params['body_pose'])
        item = {}
        # These are the keypoints in the original image coordinates (before cropping)
        #orig_keypoints_2d=[]
        #for i in range(start_index,end_index+1):
        #    orig_keypoints_2d.append(self.keypoints_2d[i].copy())
        orig_keypoints_2d=np.array(self.keypoints_2d[start_index:end_index + 1].copy())
        if (not self.cfg.MODE.EXTRACT) and (not self.cfg.MODE.IMG):
            item['features']= np.array(features)
        item['img'] = img_patch
        #print(img_patch.shape)
        item['keypoints_2d'] = keypoints_2d.astype(np.float32)
        item['keypoints_3d'] = keypoints_3d.astype(np.float32)
        item['orig_keypoints_2d'] = orig_keypoints_2d
        #import ipdb
        #ipdb.set_trace()
        item['box_center'] =self.center[start_index:end_index + 1].copy()
        item['box_size'] = self.scale[start_index:end_index + 1] * 200
        item['img_size'] = 1.0 * img_size[::-1].copy()
        item['smpl_params'] = smpl_params
        item['has_smpl_params'] = has_smpl_params
        item['smpl_params_is_axis_angle'] = smpl_params_is_axis_angle
        if self.cfg.MODE.TCMR:
            if True:
                item['keypoints_2d'] = torch.tensor(keypoints_2d.astype(np.float32)[int((end_index + 1-start_index)/2)].reshape(1,44,3))#.repeat(3,1,1)
                item['keypoints_3d'] = torch.tensor(keypoints_3d.astype(np.float32)[int((end_index + 1-start_index)/2)].reshape(1,44,4))#.repeat(3,1,1)
                item['orig_keypoints_2d'] = torch.tensor(orig_keypoints_2d[int((end_index + 1-start_index)/2)].reshape(1,44,3))#.repeat(3,1,1)
                item['box_center'] = torch.tensor(self.center[start_index:end_index + 1].copy()[int((end_index + 1-start_index)/2)].reshape(1,2))#.repeat(3,1)
                item['box_size'] = torch.tensor(self.scale[start_index:end_index + 1][int((end_index + 1-start_index)/2)])*200#.repeat(3) * 200
                item['img_size'] = 1.0 * torch.tensor(img_size[::-1].copy()[int((end_index + 1-start_index)/2)].reshape(1,2))#.repeat(3,1)
                for k in smpl_params.keys():
                    item['smpl_params'][k] = torch.tensor(smpl_params[k][int((end_index + 1-start_index)/2)].reshape(1,-1))#.repeat(3,1)
                    item['has_smpl_params'][k] = torch.tensor(has_smpl_params[k][int((end_index + 1-start_index)/2)].reshape(1,-1))#.repeat(3,1)
                item['smpl_params_is_axis_angle'] = smpl_params_is_axis_angle
            
            if not self.train:
                ins=0
                item['keypoints_2d'] = item['keypoints_2d'][ins]
                item['keypoints_3d'] = item['keypoints_3d'][ins]
                item['orig_keypoints_2d'] = item['orig_keypoints_2d'][ins]
                item['box_center'] = item['box_center']#[ins]
                item['box_size'] = item['box_size']#[ins]
                item['img_size'] = item['img_size'][ins]
                for k in smpl_params.keys():
                    item['smpl_params'][k] = item['smpl_params'][k][ins]
                    item['has_smpl_params'][k] = item['has_smpl_params'][k][ins]
                item['smpl_params_is_axis_angle'] = item['smpl_params_is_axis_angle']
        names = []
        str_img_file= ([str(l) for l in image_file])
        item['imgname']=(str_img_file)
        l=['smpl_params_is_axis_angle','features']
        #import ipdb
        #ipdb.set_trace()
        '''
        if item['features'].shape[0]==1:
            for k in item:
                if k in l:
                    continue
                if isinstance(item[k],list):
                    item[k]=[tmp for tmp in item[k] for i in range(16)]
                    continue
                if not isinstance(item[k], dict):
                    try:
                        if  item[k].numel()==1:
                            continue
                    except:
                        if item[k].ndim==0:
                            continue
                    if item[k].shape[0]!=1:
                        listofints = [int(x) for x in item[k].shape]
                        listofints.insert(0,1)
                        try:
                            item[k]=item[k].reshape(listofints).repeat_interleave(16,dim=0)
                        except:
                            item[k]=item[k].repeat(16,axis=0)

                    else:
                        item[k]=item[k].repeat(16,axis=0)
                else:
                    #print(k)
                    #print(item[k].shape)
                    for v in item[k]:
                        try:
                            item[k][v]=item[k][v].repeat(16,axis=0)
                        except:
                            item[k][v]=item[k][v].repeat_interleave(16,dim=0)
        if item['keypoints_2d'].shape[0]==44:
            import ipdb
            ipdb.set_trace()
        '''
        return item
