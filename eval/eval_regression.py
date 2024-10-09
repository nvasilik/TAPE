"""
Script used for evaluating the 3D pose errors of TAPE (mode + minimum).

Example usage:
python eval_regression.py --checkpoint=/path/to/checkpoint --dataset=3DPW-TEST

Running the above will compute the Reconstruction Error for the mode as well as the minimum error for the test set of 3DPW.
"""
import torch
import argparse
from tqdm import tqdm
from tape.utils.geometry import aa_to_rotmat, batch_adv_disc_l2_loss, batch_encoder_disc_l2_loss, perspective_projection
from tape.utils.geometry import rot6d_to_rotmat,rotation_matrix_to_angle_axis

from tape.configs import get_config, tape_config, dataset_config
from tape.models import TAPE
from tape.utils import Evaluator, recursive_to
from tape.datasets import create_dataset
from lib.dataset.loaders import get_data_loaders
from tape.utils.renderer import Renderer
from lib.data_utils.img_utils import read_image
import numpy as np
import os
import cv2
from tape.optimization import KeypointFitting
parser = argparse.ArgumentParser(description='Evaluate trained models')
parser.add_argument('--checkpoint', type=str, default='data/_checkpoint.pt', help='Path to pretrained model checkpoint')
parser.add_argument('--model_cfg', type=str, default=None, help='Path to config file. If not set use the default (tape/configs/tape.yaml)')
parser.add_argument('--dataset', type=str, default='H36M-VAL-P2', choices=['H36M-TRAIN','EGO','H36M-VAL','H36M-VAL-P2', 'H36M-VAL-P2-OPENPOSE', '3DPW-TEST','3DPW-TRAIN','3DPW-VAL','MPI-INF-TRAIN','MPI-INF_TEST'], help='Dataset to evaluate')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for inference')
parser.add_argument('--num_samples', type=int, default=1, help='Number of test samples to draw')
parser.add_argument('--num_workers', type=int, default=8, help='Number of workers used for data loading')
parser.add_argument('--log_freq', type=int, default=10, help='How often to log results')
parser.add_argument('--shuffle', dest='shuffle', action='store_true', default=False, help='Shuffle the dataset during evaluation')
parser.add_argument('--feats_file_name',type=str, help='Feats File name')


args = parser.parse_args()

# Use the GPU if available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load model config
if args.model_cfg is None:
    model_cfg = tape_config()
else:
    model_cfg = get_config(args.model_cfg)

# Load dataset config
dataset_cfg = dataset_config()[args.dataset]

# Update number of test samples drawn to the desired value
model_cfg.defrost()
model_cfg.TRAIN.NUM_TEST_SAMPLES = args.num_samples
model_cfg.freeze()

# Setup model
checkpoint = torch.load( 'data/_checkpoint.pt')
checkpoint2 = torch.load( args.checkpoint)

model = TAPE.load_from_checkpoint(args.checkpoint, strict=False, cfg=model_cfg).to(device)

model.load_state_dict(checkpoint['state_dict'],strict=False) 
model.load_state_dict(checkpoint2['state_dict'],strict=False)
model.eval()

# Create dataset and data loader
dataset = create_dataset(model_cfg, dataset_cfg, train=False)
dataloader = torch.utils.data.DataLoader(dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)
#t_dataloader,v_dataloader = get_data_loaders(None)
#dataloader=v_dataloader
# List of metrics to log
# Create a dataset on-the-fly

#from tape.datasets.openpose_dataset import OpenPoseDataset
#dataset = OpenPoseDataset(model_cfg, img_folder="example_data/images", keypoint_folder="example_data/keypoints", max_people_per_image=1)

# Setup a dataloader with batch_size = 1 (Process images sequentially)
metrics = ['mode_re', 'mode_mpjpe','min_re']#,'mpvpe']#, 'min_re','accel_err']#,

# Setup evaluator object
evaluator = Evaluator(dataset_length=len(dataset), keypoint_list=dataset_cfg.KEYPOINT_LIST, pelvis_ind=model_cfg.EXTRA.PELVIS_IND, metrics=metrics)
feats=[]
renderer = Renderer(model_cfg, faces=model.smpl.faces)
# Go over the images in the dataset.
#keypoint_fitting = KeypointFitting(model_cfg)

#count=1
#out_folder='./outH36m'
for i, batch in enumerate(tqdm(dataloader)):
    batch = recursive_to(batch, device)
    #count=count+1
    with torch.no_grad():
        out = model(batch)
        batch_size = batch['img'].shape[0]
    
    #for n in range(batch_size):
        #img_fn, _ = os.path.splitext(os.path.split(batch['imgname'][n])[1])
        #import ipdb
        #ipdb.set_trace()

        #pred_cam=out['pred_cam'][-1,0]
        # box_center = batch['box_center'].reshape(1,2)
        #box_size = batch['box_size']
        #img_size = batch['img_size']
        #tmp = img_size[0,0].detach().clone()
        #img_size[0,0]=img_size[0,1]
        #img_size[0,1]=tmp
            #import ipdb
            #ipdb.set_trace()
        #camera_center = 0.5 * img_size
        #depth = 2 * 5000 / (box_size.reshape(batch_size, 1) * pred_cam[0].reshape(batch_size, 1) + 1e-9)
        #init_cam_t = torch.zeros_like(pred_cam)
        #init_cam_t[ :2] = pred_cam[1:] + (box_center - camera_center) * depth / 5000
        #init_cam_t[ -1] = depth.reshape(batch_size)
        #regression_img = renderer(out['pred_vertices'][-1, 0].detach().cpu().numpy(),
        #                            init_cam_t.detach().cpu().numpy(),
        #                          #out['pred_cam_t'][-1, 0].detach().cpu().numpy(),
        #                            batch['img'][-1].reshape(-1,3,224,224)[8],bscolor=(1,1,0.9,1), imgname=batch['imgname'][8], full_frame=True)

                                    #batch['img'][-1].reshape(-1,3,224,224)[8])
                                  #batch['img'][-1])
        #out_str='reg'+str(i)+'.jpg'
        #cv2.imwrite(os.path.join(out_folder, out_str), 255*regression_img[:, :, ::-1])

        #import ipdb
        #ipdb.set_trace()
        #cv2.imwrite(os.path.join(args.out_folder, f'{img_fn}_regression.{args.out_format}'), 255*regression_img[:, :, ::-1])
    if False:
        opt_out = model.downstream_optimization(regression_output=out,
                                                batch=batch,
                                                opt_task=keypoint_fitting,
                                                use_hips=False,
                                                full_frame=True)
        batch['imgname']=np.array(batch['imgname'])
        for n in range(batch_size):
            img_fn, _ = os.path.splitext(os.path.split(batch['imgname'][0][-1])[1])
            fitting_img = renderer(opt_out['vertices'][n].detach().cpu().numpy(),
                                   opt_out['camera_translation'][n].detach().cpu().numpy(),
                                   batch['img'][-1].reshape(-1,3,224,224)[8],bscolor=(0.1,0.7,0.9,1.0), imgname=batch['imgname'][8], full_frame=True)
            cv2.imwrite(os.path.join(out_folder, f'{img_fn}_fitting.jpg'), 255*fitting_img[:, :, ::-1])
    '''
    
    path=batch['imgname'][-1][-1]
    or_im = read_image(path)
    
    regression_img = renderer(out['pred_vertices'][-1, -1].detach().cpu().numpy(),
                                    out['pred_cam_t'][-1,-1].detach().cpu().numpy(),                                        
                                    batch['img'][-1].reshape(16,3,224,224)[8])
    outfile='video_out/'+str(count)+'.jpg'
    cv2.imwrite(outfile,255*regression_img[:,:,::-1])
    import ipdb
    ipdb.set_trace()
    count=count+1
    '''

    #i=i+500
    '''
    
    
    
    #keypoint_fitting = KeypointFitting(model_cfg)
    
    #    
    #opt_out = model.downstream_optimization(regression_output=out,
    #                                            batch=batch,
    #                                            opt_task=keypoint_fitting,
    #                                            use_hips=False,
    #                                            full_frame=True)
    #fitting_img = renderer(opt_out['vertices'][-1].detach().cpu().numpy(),
    #                               opt_out['camera_translation'][-1].detach().cpu().numpy(),
    #                               batch['img'][-1].reshape(3,224,224),full_frame=True, imgname='example_data/images/drosos.jpg')
    #cv2.imwrite('drosos_opt.jpg',255*fitting_img[:,:,::-1])
    #import ipdb
    #ipdb.set_trace()
    '''

    '''
    if count<301:
        path=batch['imgname'][-1][-1]
        or_im = read_image(path)
        #regression_img = renderer(out['pred_vertices'][-1, -1].detach().cpu().numpy(),
        #                          out['pred_cam_t'][-1, -1].detach().cpu().numpy(),
        #                          batch['img'][-1].reshape(3,224,224))#,full_frame=True,imgname=path)
        #import ipdb
        #ipdb.set_trace()
        pred_cam=out['pred_cam'][-1,-1].detach().cpu().numpy()
        box_center = batch['box_center'][-1,-1].detach().cpu().numpy()
        box_size = batch['box_size'][-1,-1].detach().cpu().numpy()
        img_size = batch['img_size'][-1,-1]

        camera_center = box_center
        focal_length=5000*box_size/224
        #depth = focal_length / (box_size.reshape(1, 1) * pred_cam[0].reshape(1, 1) + 1e-9)
        #init_cam_t = torch.zeros_like(pred_cam)
        #init_cam_t[:2] = pred_cam[1:] +  depth / focal_length
        #init_cam_t[-1] = depth.reshape(1)
        cx, cy, h = 224, 224, box_size
        hw, hh = 1002 / 2., 1000 / 2.
        sx = pred_cam[0] * (1. / (1002 / h))
        sy = pred_cam[0] * (1. / (1000 / h))
        tx = ((cx - hw) / hw / sx) + pred_cam[1]
        ty = ((cy - hh) / hh / sy) + pred_cam[2]
        orig_cam = torch.tensor(np.stack([sx, sy, tx, ty]).T)
        pred_cam_t = torch.stack([
        orig_cam[1],orig_cam[2],
        2 * 5000 / (224 * orig_cam[0] + 1e-9)
        ], dim=-1)
        regression_img = renderer(out['pred_vertices'][-1, -1].detach().cpu().numpy(),
                                pred_cam_t.detach().cpu().numpy(),
                                or_im,full_frame = True,imgname=path)
        #outfile='video_out/'+str(count)+'.jpg'
        cv2.imwrite('fullframe.jpg',255*regression_img[:,:,::-1])
        import ipdb
        ipdb.set_trace()
        keypoint_fitting = KeypointFitting(model_cfg)
        
        opt_out = model.downstream_optimization(regression_output=out,
                                                batch=batch,
                                                opt_task=keypoint_fitting,
                                                use_hips=False,
                                                full_frame=True)
        fitting_img = renderer(opt_out['vertices'][-1].detach().cpu().numpy(),
                                   opt_out['camera_translation'][-1].detach().cpu().numpy(),
                                   batch['img'][-1].reshape(3,224,224),full_frame=True, imgname=path)
        #outfile='video_out/opt0_'+str(count)+'.jpg'
        #cv2.imwrite(outfile,255*regression_img[:,:,::-1])

        #fitting_img = renderer(opt_out['vertices'][-1].detach().cpu().numpy(),
        #                           opt_out['camera_translation'][-1].detach().cpu().numpy(),
        #                           batch['img'][-1].reshape(or_im.shape))#,full_frame=True, imgname=path)
        #outfile='video_out/opt0_'+str(count)+'.jpg'
        #cv2.imwrite(outfile,255*regression_img[:,:,::-1])
        
        #import ipdb
        #ipdb.set_trace()
    
    count=count+1 
    '''
    #for k in range(out['conditioning_feats'].shape[0]):
    #    feats.append(out['conditioning_feats'][k].cpu().detach().numpy())
    import ipdb
    ipdb.set_trace()
    evaluator(out, batch)
    if i % args.log_freq == args.log_freq - 1:
        evaluator.log()
#np.savez(args.feats_file_name,features=feats)