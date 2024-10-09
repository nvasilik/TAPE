import os
from socket import IP_DROP_MEMBERSHIP
import sys
from tkinter import E
import cv2
import glob
import h5py
import json
import numpy as np
import scipy.io as sio
import scipy.misc
import imageio

def read_calibration(calib_file, vid_list):
    Ks, Rs, Ts = [], [], []
    file = open(calib_file, 'r')
    content = file.readlines()
    for vid_i in vid_list:
        K = np.array([float(s) for s in content[vid_i*7+5][11:-2].split()])
        K = np.reshape(K, (4, 4))
        RT = np.array([float(s) for s in content[vid_i*7+6][11:-2].split()])
        RT = np.reshape(RT, (4, 4))
        R = RT[:3,:3]
        T = RT[:3,3]/1000
        Ks.append(K)
        Rs.append(R)
        Ts.append(T)
    return Ks, Rs, Ts
    
def train_data(dataset_path, out_path, joints_idx, scaleFactor, extract_img=False, fits_3d=None):

    joints17_idx = [4, 18, 19, 20, 23, 24, 25, 3, 5, 6, 7, 9, 10, 11, 14, 15, 16]

    h, w = 2048, 2048
    imgnames_, scales_, centers_ = [], [], []
    parts_, Ss_,  = [], []
    extra_keypoints_3d_,extra_keypoints_2d_=[],[]
    body_pose_,betas_,has_body_pose_,has_betas_=[],[],[],[]
    # training data
    user_list = range(1,9)
    seq_list = range(1,3)
    vid_list = list(range(3)) + list(range(4,9))

    counter = 0
    fits = np.load('./data/static_fits/mpi-inf-3dhp_mview_fits.npz')
    poses=fits['pose']
    betas=fits['shape']
    for user_i in user_list:
        for seq_i in seq_list:
            seq_path = os.path.join(dataset_path,
                                    'S' + str(user_i),
                                    'Seq' + str(seq_i))
            # mat file with annotations
            annot_file = os.path.join(seq_path, 'annot.mat')
            annot2 = sio.loadmat(annot_file)['annot2']
            annot3 = sio.loadmat(annot_file)['annot3']
            # calibration file and camera parameters
            calib_file = os.path.join(seq_path, 'camera.calibration')
            Ks, Rs, Ts = read_calibration(calib_file, vid_list)

            for j, vid_i in enumerate(vid_list):

                # image folder
                imgs_path = os.path.join(seq_path,    
                                         'imageFrames',
                                         'video_' + str(vid_i))

                # extract frames from video file
                if extract_img:

                    # if doesn't exist
                    if not os.path.isdir(imgs_path):
                        os.makedirs(imgs_path)

                    # video file
                    vid_file = os.path.join(seq_path,
                                            'imageSequence',
                                            'video_' + str(vid_i) + '.avi')
                    vidcap = cv2.VideoCapture(vid_file)

                    # process video
                    frame = 0
                    while 1:
                        # extract all frames
                        success, image = vidcap.read()
                        if not success:
                            break
                        frame += 1
                        # image name
                        imgname = os.path.join(imgs_path,
                            'frame_%06d.jpg' % frame)
                        # save image
                        cv2.imwrite(imgname, image)

                # per frame
                cam_aa = cv2.Rodrigues(Rs[j])[0].T[0]
                pattern = os.path.join(imgs_path, '*.jpg')
                img_list = sorted(glob.glob(pattern))
                for i, img_i in enumerate(img_list):
                    # for each image we store the relevant annotations
                    img_name = img_i.split('/')[-1]
                    img_view = os.path.join('S' + str(user_i),
                                            'Seq' + str(seq_i),
                                            'imageFrames',
                                            'video_' + str(vid_i),
                                            img_name)
                    joints = np.reshape(annot2[vid_i][0][i], (28, 2))[joints17_idx]
                    S17 = np.reshape(annot3[vid_i][0][i], (28, 3))/1000
                    S17 = S17[joints17_idx] - S17[4] # 4 is the root
                    bbox = [min(joints[:,0]), min(joints[:,1]),
                            max(joints[:,0]), max(joints[:,1])]
                    center = [(bbox[2]+bbox[0])/2, (bbox[3]+bbox[1])/2]
                    scale = scaleFactor*max(bbox[2]-bbox[0], bbox[3]-bbox[1])

                    # check that all joints are visible
                    x_in = np.logical_and(joints[:, 0] < w, joints[:, 0] >= 0)
                    y_in = np.logical_and(joints[:, 1] < h, joints[:, 1] >= 0)
                    ok_pts = np.logical_and(x_in, y_in)
                    if np.sum(ok_pts) < len(joints_idx):
                        continue
                        
                    part = np.zeros([19,3])
                    part[joints_idx] = np.hstack([joints, np.ones([17,1])])
                    extra_keypoints_2d = part
                    tmp=extra_keypoints_2d[18].copy()
                    extra_keypoints_2d[18]=extra_keypoints_2d[13]
                    extra_keypoints_2d[13]=tmp
                    #json_file = os.path.join(openpose_path, 'mpi_inf_3dhp',
                    #    img_view.replace('.jpg', '_keypoints.json'))
                    #openpose = read_openpose(json_file, part, 'mpi_inf_3dhp')

                    S = np.zeros([19,4])
                    S[joints_idx] = np.hstack([S17, np.ones([17,1])])
                    extra_keypoints_3d = S
                    tmp=extra_keypoints_3d[18].copy()
                    extra_keypoints_3d[18]=extra_keypoints_3d[13]
                    extra_keypoints_3d[13]=tmp

                    # store the data
                    extra_keypoints_3d_.append(extra_keypoints_3d)
                    extra_keypoints_2d_.append(extra_keypoints_2d)

                    # because of the dataset size, we only keep every 10th frame
                    #if counter % 10 != 1:
                    #    continue
                    #import ipdb
                    #ipdb.set_trace()
                    if counter%10 ==1:
                        body_pose_.append(poses[int(counter/10)])
                        betas_.append(betas[int(counter/10)])
                        has_body_pose_.append(1)
                        has_betas_.append(1)
                        counter+=1
                    else:
                        body_pose_.append(poses[int(counter)])
                        betas_.append(betas[int(counter)])
                        has_body_pose_.append(0)
                        has_betas_.append(0)
                    #counter += 1
                    # store the data
                    imgnames_.append(img_view)
                    centers_.append(center)
                    scales_.append(scale)
                    #parts_.append(part)
                    #Ss_.append(S)
                    #openposes_.append(openpose)
                       
    # store the data struct
    print('1')
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    out_file = os.path.join(out_path, 'mpi_train_smpl.npz')
    if fits_3d is not None:
        #fits_3d = np.load(fits_3d)
        np.savez(out_file, imgname=imgnames_,
                           center=centers_,
                           scale=scales_,
                           #part=parts_,
                           body_pose=np.array(body_pose_),
                           betas=np.array(betas_),
                           has_body_pose=has_body_pose_,
                           has_betas=has_betas_,
                           #has_smpl_params=fits_3d['has_smpl'],
                           #S=Ss_,
                           #has_body_pose=np.ones(len(fits_3d['pose'])),
                           #has_betas=np.ones(len(fits_3d['shape'])),
                           extra_keypoints_2d=extra_keypoints_2d_,
                           extra_keypoints_3d=extra_keypoints_3d_)   
                           #openpose=openposes_)
        print('2')
    else:
        np.savez(out_file, imgname=imgnames_,
                           center=centers_,
                           scale=scales_,
                           part=parts_,
                           S=Ss_,)
                           #openpose=openposes_)        
        
        
def test_data(dataset_path, out_path, joints_idx, scaleFactor):

    joints17_idx = [14, 11, 12, 13, 8, 9, 10, 15, 1, 16, 0, 5, 6, 7, 2, 3, 4]
    global_idx=joints_idx
    imgnames_, scales_, centers_, parts_,  Ss_,extra_keypoints_2d_,extra_keypoints_3d_ = [], [], [], [], [], [], []

    # training data
    user_list = range(1,7)

    for user_i in user_list:
        seq_path = os.path.join(dataset_path,
                                'mpi_inf_3dhp_test_set',
                                'TS' + str(user_i))
        # mat file with annotations
        annot_file = os.path.join(seq_path, 'annot_data.mat')
        mat_as_h5 = h5py.File(annot_file, 'r')
        annot2 = np.array(mat_as_h5['annot2'])
        annot3 = np.array(mat_as_h5['univ_annot3'])
        valid = np.array(mat_as_h5['valid_frame'])
        for frame_i, valid_i in enumerate(valid):
            #if valid_i == 0:
                #continue
            img_name = os.path.join('mpi_inf_3dhp_test_set',
                                   'TS' + str(user_i),
                                   'imageSequence',
                                   'img_' + str(frame_i+1).zfill(6) + '.jpg')

            joints = annot2[frame_i,0,joints17_idx,:]
            S17 = annot3[frame_i,0,joints17_idx,:]/1000
            S17 = S17 - S17[0]


            bbox = [min(joints[:,0]), min(joints[:,1]),
                    max(joints[:,0]), max(joints[:,1])]
            center = [(bbox[2]+bbox[0])/2, (bbox[3]+bbox[1])/2]
            scale = scaleFactor*max(bbox[2]-bbox[0], bbox[3]-bbox[1])

            # check that all joints are visible
            img_file = os.path.join(dataset_path, img_name)
            I = imageio.imread(img_file)
            h, w, _ = I.shape
            x_in = np.logical_and(joints[:, 0] < w, joints[:, 0] >= 0)
            y_in = np.logical_and(joints[:, 1] < h, joints[:, 1] >= 0)
            ok_pts = np.logical_and(x_in, y_in)
            #if np.sum(ok_pts) < len(joints_idx):
            #    continue

            part = np.zeros([19,3])
            part[joints17_idx] = np.hstack([joints, np.ones([17,1])])
            #part17=part[joints17_idx]
            extra_keypoints_2d = part
            tmp=extra_keypoints_2d[18].copy()
            extra_keypoints_2d[18]=extra_keypoints_2d[13]
            extra_keypoints_2d[13]=tmp
            #extra_keypoints_2d[global_idx,:] = part17
            #extra_keypoints_2d[global_idx, 2] = 1

            S = np.zeros([19,4])
            S[joints_idx] = np.hstack([S17, np.ones([17,1])])
            extra_keypoints_3d = S
            tmp=extra_keypoints_3d[18].copy()
            extra_keypoints_3d[18]=extra_keypoints_3d[13]
            extra_keypoints_3d[13]=tmp

            # store the data
            extra_keypoints_3d_.append(extra_keypoints_3d)
            extra_keypoints_2d_.append(extra_keypoints_2d)
            imgnames_.append(img_name)
            centers_.append(center)
            scales_.append(scale)
            parts_.append(part)
            Ss_.append(S)

    # store the data struct
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    out_file = os.path.join(out_path, 'mpi_inf_3dhp_valid.npz')
    np.savez(out_file, imgname=imgnames_,
                       center=centers_,
                       scale=scales_,
                       part=parts_,
                       S=Ss_,
                       extra_keypoints_2d=extra_keypoints_2d_,
                       extra_keypoints_3d=extra_keypoints_3d_)    

def mpi_inf_3dhp_extract(dataset_path, out_path, mode, extract_img=False, static_fits=None):

    scaleFactor = 1.2
    joints_idx = [14, 3, 4, 5, 2, 1, 0, 16, 12, 17, 18, 9, 10, 11, 8, 7, 6]
    
    if static_fits is not None:
        fits_3d = os.path.join(static_fits, 
                               'mpi-inf-3dhp_mview_fits.npz')
    else:
        fits_3d = None

    if mode == 'train':
        train_data(dataset_path, out_path, 
                   joints_idx, scaleFactor, extract_img=extract_img, fits_3d=fits_3d)
    elif mode == 'test':
        test_data(dataset_path, out_path, joints_idx, scaleFactor)
mpi_inf_3dhp_extract('/media/nikos/Samsung1TB/mpi-3dhp/mpi_inf_3dhp','./data', 'train', extract_img=False, static_fits='./data/static_fits')
#mpi_inf_3dhp_extract('/media/nikos/Samsung1TB/mpi-3dhp/mpi_inf_3dhp/mpi_inf_3dhp_test_set','./data', 'test', extract_img=False, static_fits=None)