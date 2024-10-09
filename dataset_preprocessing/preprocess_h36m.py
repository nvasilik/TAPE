import os
import sys
from this import d
import cv2
import glob
import h5py
import numpy as np
import argparse
os.environ["CDF_LIB"] = "/home/nikos/cdf38_1-dist/" # Give the path of the cdf library
from spacepy import pycdf
import pickle as pkl
import pickle
import pandas as pd
import torch
from tape.configs import tape_config, dataset_config
from lib.data_utils.amass_utils import joints_to_use
#from tape.utils.geometry import batch_rodrigues
from lib.utils.geometry import rotation_matrix_to_angle_axis,batch_rodrigues
parser = argparse.ArgumentParser(description='Generate H36M dataset files')
parser.add_argument('--split', type=str, required=True, choices=['VAL', 'VAL-P2', 'TRAIN', 'MULTIVIEW'], help='Dataset split to preprocess')
joints_to_use = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 37])
joints_to_use = np.arange(0,156).reshape((-1,3))[joints_to_use].reshape(-1)

args = parser.parse_args()

def read_pkl(f):
    return pkl.load(open(f, 'rb'), encoding='latin1')
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
def double_rectify(pose):
    """
    Rectify "upside down" people in global coord
 
    Args:
        pose (72,): Pose.

    Returns:
        Rotated pose.
    """
    pose = pose.copy()
    R_mod = cv2.Rodrigues(np.array([2*np.pi, 0, 0]))[0]
    R_root = cv2.Rodrigues(pose[:3])[0]
    new_root = R_root.dot(R_mod)
    pose[:3] = cv2.Rodrigues(new_root)[0].reshape(3)
    return pose
def rectify_pose(camera_r, body_aa):
    body_r = batch_rodrigues(body_aa).reshape(-1,3,3)
    final_r = camera_r @ body_r
    body_aa = rotation_matrix_to_angle_axis(final_r)
    return body_aa
def load_camera_params( hf, path ):
    """Load h36m camera parameters
    Args
    hf: hdf5 open file with h36m cameras data
    path: path or key inside hf to the camera we are interested in
    Returns
    R: 3x3 Camera rotation matrix
    T: 3x1 Camera translation parameters
    f: (scalar) Camera focal length
    c: 2x1 Camera center
    k: 3x1 Camera radial distortion coefficients
    p: 2x1 Camera tangential distortion coefficients
    name: String with camera id
    """

    R = hf[ path.format('R') ][:]
    R = R.T

    T = hf[ path.format('T') ][:]
    f = hf[ path.format('f') ][:]
    c = hf[ path.format('c') ][:]
    k = hf[ path.format('k') ][:]
    p = hf[ path.format('p') ][:]

    name = hf[ path.format('Name') ][:]
    name = "".join( [chr(item) for item in name] )

    return R, T, f, c, k, p, name


def load_cameras( bpath='/media/nikos/Samsung1TB/h36m/train/cameras.h5', subjects=[1,5,6,7,8,9,11]):
    """Loads the cameras of h36m
    Args
    bpath: path to hdf5 file with h36m camera data
    subjects: List of ints representing the subject IDs for which cameras are requested
    Returns
    rcams: dictionary of 4 tuples per subject ID containing its camera parameters for the 4 h36m cams
    """
    rcams = {}

    with h5py.File(bpath,'r') as hf:
        for s in subjects:
            for c in range(4): # There are 4 cameras in human3.6m
                rcams[c+1] = load_camera_params(hf, 'subject%d/camera%d/{0}' % (s,c+1) )

    return rcams
def read_mosh_data(f, frame_skip=4, cam=None):

    data = read_pkl(f)
    poses = data['poses'][0::frame_skip]
    shape = data['betas'][:10]
    shape = np.tile(shape, [poses.shape[0], 1])

    camera_ids = {
        '54138969': 1,
        '55011271': 2,
        '58860488': 3,
        '60457274': 4,
    }

    rcams = load_cameras()
    camR = rcams[camera_ids[cam]][0]

    poses = torch.from_numpy(poses)
    camR = torch.from_numpy(camR).unsqueeze(0)

    poses[:,:3] = rectify_pose(camR, poses[:,:3])

    return poses.numpy(), shape


def preprocess_h36m(dataset_path: str, out_file: str, split: str, extract_img: bool = False,mosh: bool=True):
    '''
    Generate H36M training and validation npz files
    Args:
        dataset_path (str): Path to H36M root
        out_file (str): Output filename
        split (str): Whether it is TRAIN/VAL/VAL-P2
        extract_img: Whether to extract the images from the videos
    '''

    # convert joints to global order
    h36m_idx = [11, 6, 7, 8, 1, 2, 3, 12, 24, 14, 15, 17, 18, 19, 25, 26, 27]
    global_idx = [14, 3, 4, 5, 2, 1, 0, 16, 12, 17, 18, 9, 10, 11, 8, 7, 6]

    # structs we use
    imgnames_, scales_, centers_, extra_keypoints_2d_, extra_keypoints_3d_,body_pose_,betas_  = [], [], [], [], [],[],[]

    if split == 'TRAIN':
        user_list = [1, 5, 6, 7, 8]
    elif split == 'VAL' or split == 'VAL-P2':
        user_list = [9, 11]
    # go over each user
    for user_i in user_list:
        user_name = 'S%d' % user_i
        # path with GT bounding boxes
        bbox_path = os.path.join(dataset_path, user_name, 'MySegmentsMat', 'ground_truth_bb')
        #print(dataset_path)
        #print(user_name)
        # path with GT 3D pose
        pose_path = os.path.join(dataset_path, user_name, 'MyPoseFeatures', 'D3_Positions_mono')
        # path with GT 2D pose
        pose2d_path = os.path.join(dataset_path, user_name, 'MyPoseFeatures', 'D2_Positions')
        # path with videos
        vid_path = os.path.join(dataset_path, user_name, 'Videos')

        #
        if mosh:
            mosh_path = os.path.join(dataset_path, 'mosh', user_name)
        # go over all the sequences of each user
        seq_list = glob.glob(os.path.join(pose_path, '*.cdf'))
        seq_list.sort()
        #print(seq_list)
        for seq_i in seq_list:

            # sequence info
            seq_name = seq_i.split('/')[-1]
            action, camera, _ = seq_name.split('.')
            action = action.replace(' ', '_')
            action.split('_')
            # irrelevant sequences
            if action == '_ALL':
                continue

            # 3D pose file
            poses_3d = pycdf.CDF(seq_i)['Pose'][0]

            # 2D pose file
            pose2d_file = os.path.join(pose2d_path, seq_name)
            poses_2d = pycdf.CDF(pose2d_file)['Pose'][0]

            # bbox file
            bbox_file = os.path.join(bbox_path, seq_name.replace('cdf', 'mat'))
            bbox_h5py = h5py.File(bbox_file)

            # video file
            if extract_img:
                vid_file = os.path.join(vid_path, seq_name.replace('cdf', 'mp4'))
                imgs_path = os.path.join(dataset_path, 'images')#CAPS
                vidcap = cv2.VideoCapture(vid_file)
                #success, image = vidcap.read()
            
            
            frame_skip=2
            if user_i not in [9, 11] and mosh:

                camera_ids = {
                    '54138969': 0,
                    '55011271': 1,
                    '58860488': 2,
                    '60457274': 3,
                }
                camera_idxed='%s%s%s%s' % (f'{action.replace("_", " ")}','_cam',f'{camera_ids[camera]}','_aligned.pkl')
                #mosh_file = os.path.join(mosh_path, camera_idxed)
                #data = read_pkl(mosh_file)
                #betas2=data['betas']
                #poses2=data['new_poses']
                
                mosh_file = os.path.join(mosh_path, f'{action.replace("_", " ")}.pkl')
                poses,betas=read_mosh_data(mosh_file, frame_skip=4, cam=camera)
                #data = read_pkl(mosh_file)
                #betas=data['shape']
                #poses=data['pose']
            # go over each frame of the sequence
            for frame_i in range(0,poses_3d.shape[0],frame_skip):
                # read video frame
                if extract_img:
                    success, image = vidcap.read()
                    if not success:
                        break

                # check if you can keep this frame
                if frame_i % 1 == 0 and (split == 'VAL' or split == 'TRAIN' or camera == '60457274'):
                    # image name
                    imgname = '%s_%s.%s_%06d.jpg' % (user_name, action, camera, frame_i+1)
                    # save image
                    if extract_img:
                        img_out = os.path.join(imgs_path, imgname)
                        #print(img_out)
                        cv2.imwrite(img_out, image)
                    # read GT 2D pose
                    partall = np.reshape(poses_2d[frame_i,:], [-1,2])
                    part17 = partall[h36m_idx]
                    extra_keypoints_2d = np.zeros([19,3])
                    extra_keypoints_2d[global_idx, :2] = part17
                    extra_keypoints_2d[global_idx, 2] = 1

                    # read GT 3D pose
                    Sall = np.reshape(poses_3d[frame_i,:], [-1,3])/1000.
                    S17 = Sall[h36m_idx]
                    S17 -= S17[0] # root-centered
                    extra_keypoints_3d = np.zeros([19,4])
                    extra_keypoints_3d[global_idx, :3] = S17
                    extra_keypoints_3d[global_idx, 3] = 1
                    # read GT bounding box
                    #if imgname == 'S9_Directions_1.58860488_000001.jpg':
                    #    import ipdb
                    #    ipdb.set_trace()
                    mask = bbox_h5py[bbox_h5py['Masks'][frame_i,0]].value.T
                    ys, xs = np.where(mask==1)
                    bbox = np.array([np.min(xs), np.min(ys), np.max(xs)+1, np.max(ys)+1])
                    center = [(bbox[2]+bbox[0])/2, (bbox[3]+bbox[1])/2]
                    scale = 0.9*max(bbox[2]-bbox[0], bbox[3]-bbox[1])

                    if user_i not in [9, 11] and mosh:
                        #body_pose_f=poses[int(frame_i/5)]
                        try:
                            body_pose_f=double_rectify(poses[int(frame_i)])
                            betas_f=betas[int(frame_i)]
                        except:
                            print('Real Shape :')
                            print(poses_3d.shape)
                            print('Betas Shape ')
                            print(betas.shape)
                            print(mosh_file)
                            continue
                            body_pose_f=poses[frame_i]
                            betas_f=betas[-1]
                        #betas_f=betas_act
                        #betas_f=betas#[frame_i]

                    # store data
                    if(mosh):
                        body_pose_.append(body_pose_f)
                        betas_.append(betas_f)
                    imgnames_.append(os.path.join('images', imgname))
                    centers_.append(center)
                    scales_.append(scale)
                    extra_keypoints_2d_.append(extra_keypoints_2d)
                    extra_keypoints_3d_.append(extra_keypoints_3d)

    # store the data struct
    #if not os.path.isdir(out_file):
        #os.makedirs(out_file)
    if mosh:
        np.savez(out_file, imgname=imgnames_,
                        center=centers_,
                        scale=scales_,
                        body_pose=body_pose_,
                        has_body_pose=np.ones(len(body_pose_)),
                        betas=betas_,
                        has_betas=np.ones(len(betas_)),
                        extra_keypoints_2d=extra_keypoints_2d_,
                        extra_keypoints_3d=extra_keypoints_3d_)
    else:
        np.savez(out_file, imgname=imgnames_,
                        center=centers_,
                        scale=scales_,
                        extra_keypoints_2d=extra_keypoints_2d_,
                        extra_keypoints_3d=extra_keypoints_3d_)

def preprocess_h36m_multiview(input_file: str, out_file: str):
    '''
    Generate H36M multiview evaluation file
    Args:
        input_file (str): H36M validation npz filename
        out_file (str): Output filename
    '''
    x = dict(np.load(input_file))
    imgname = x['imgname']
    actions = np.unique([img.split('/')[-1].split('.')[0] for img in imgname])
    frames = {action: {} for action in actions}
    for i, img in enumerate(imgname):
        action_with_cam = img.split('/')[-1]
        action = action_with_cam.split('.')[0]
        cam = action_with_cam.split('.')[1].split('_')[0]
        if cam in frames[action]:
            frames[action][cam].append(i)
        else:
            frames[action][cam] = []
    data_list = []
    for action in frames.keys():
        cams = list(frames[action].keys())
        for n in range(len(frames[action][cams[0]])):
            keep_frames = []
            for cam in cams:
                keep_frames.append(frames[action][cam][n])
            data_list.append({k: v[keep_frames] for k,v in x.items()})
    pickle.dump(data_list, open(out_file, 'wb'))

if __name__ == '__main__':
    dataset_cfg = dataset_config()[f'H36M-{args.split}']
    if args.split == 'MULTIVIEW':
        preprocess_h36m_multiview(dataset_config()['H36M-VAL'].DATASET_FILE, dataset_cfg.DATASET_FILE)
    else:
        preprocess_h36m("/media/nikos/Samsung1TB/h36m/train", "/media/nikos/Samsung1TB/h36m/train", 'TRAIN', extract_img=False,mosh=True)
        #preprocess_h36m("/media/nikos/Samsung1TB/h36m/val", "/media/nikos/Samsung1TB/h36m/val", 'VAL', extract_img=False,mosh=False)
        #preprocess_h36m("/media/nikos/Samsung1TB/h36m/val", "/media/nikos/Samsung1TB/h36m/val-p2",'VAL-P2', extract_img=False,mosh=False)


