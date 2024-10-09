"""
Script used for evaluating the 3D pose errors for fitting on top of regression.

Example usage:
python eval_keypoint_fitting.py --checkpoint=/path/to/checkpoint --dataset=3DPW-TEST

Running the above will compute the Reconstruction Error before and after fitting for the test set of 3DPW.
"""
import torch
import argparse
from tqdm import tqdm
from tape.configs import get_config, tape_config, dataset_config
from tape.models import TAPE
from tape.optimization import KeypointFitting
from tape.utils import recursive_to,Evaluator, recursive_to
from tape.datasets import create_dataset
from lib.dataset.loaders import get_data_loaders
from lib.data_utils.img_utils import get_single_image_crop
from tape.utils.render_openpose import render_openpose

from tape.utils.renderer import Renderer
from lib.data_utils.img_utils import read_image
#from lib.core.evaluate import Evaluator
import cv2
import os
import ipdb
import numpy as np

parser = argparse.ArgumentParser(description='Evaluate trained model on keypoint fitting')
parser.add_argument('--checkpoint', type=str, default='data/checkpoint.pt', help='Path to pretrained model checkpoint')
parser.add_argument('--model_cfg', type=str, default=None, help='Path to config file. If not set use the default (tape/configs/tape.yaml)')
parser.add_argument('--dataset', type=str, required=True, choices=['H36M-VAL-P2', 'H36M-VAL-P2-OPENPOSE', '3DPW-TEST','3DPW-TRAIN','MPI-INF_TEST'], help='Dataset to evaluate')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference/fitting')
parser.add_argument('--num_workers', type=int, default=12, help='Number of workers used for data loading')
parser.add_argument('--log_freq', type=int, default=1, help='How often to log results')
parser.add_argument('--shuffle', dest='shuffle', action='store_true', default=False, help='Shuffle the dataset during evaluation')
parser.add_argument('--num_samples', type=int, default=2, help='Number of test samples to draw')

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
model_cfg.defrost()
model_cfg.TRAIN.NUM_TEST_SAMPLES = args.num_samples
model_cfg.freeze()
# Setup model
model = TAPE.load_from_checkpoint(args.checkpoint, strict=False, cfg=model_cfg).to(device)
model.eval()

# Setup fitting
keypoint_fitting = KeypointFitting(model_cfg, max_iters=30)

# Create dataset and data loader
dataset = create_dataset(model_cfg, dataset_cfg, train=False)
dataloader = torch.utils.data.DataLoader(dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)
#t_dataloader,v_dataloader = get_data_loaders(None)
#dataloader=v_dataloader
# List of metrics to log
metrics = ['mode_re','opt_re','opt_mpjpe','opt_accel']#,'opt_re', 'opt_mpjpe']

# Setup evaluator object
#evaluator = Evaluator(dataloader,model).run()
evaluator = Evaluator(dataset_length=len(dataset), keypoint_list=dataset_cfg.KEYPOINT_LIST, pelvis_ind=model_cfg.EXTRA.PELVIS_IND, metrics=metrics)
#renderer = Renderer(model_cfg, faces=model.smpl.faces)
# Go over the images in the dataset.
renderer = Renderer(model_cfg, faces=model.smpl.faces)

for i, batch in enumerate(tqdm(dataloader)):
    batch = recursive_to(batch, device)
    with torch.no_grad():
       out = model(batch)
    opt_out = model.downstream_optimization(regression_output=out,
                                                batch=batch,
                                               opt_task=keypoint_fitting,
                                               use_hips=dataset_cfg.USE_HIPS,
                                               full_frame=False)
    
    #for l in range(16):
    #    path=batch['imgname'][l][-1]
    #    or_im = read_image(path)
    #    fitting_img = renderer(opt_out['vertices'][l].detach().cpu().numpy(),
    #                                opt_out['camera_translation'][l].detach().cpu().numpy(),
    #                                batch['img'][l].reshape(3,224,224),full_frame=False, imgname=path)        #outfile='video_out/opt0_'+str(count)+'.jpg'
    #    outfile='video_out/opt1_'+str(count)+'.jpg'
    #    count = count+1
    #    cv2.imwrite(outfile,255*fitting_img[:,:,::-1])

    '''
    for l in range(16):
        path=batch['imgname'][l][-1]
        or_im = read_image(path)
        regression_img = renderer(out['pred_vertices'][l, -1].detach().cpu().numpy(),
                        out['pred_cam_t'][l, -1].detach().cpu().numpy(),
                        batch['img'][l].reshape(3,224,224))#,full_frame = True,imgname=path)
        outfile='video_out/'+str(count2)+'.jpg'
        count2=count2+1
        cv2.imwrite(outfile,255*regression_img[:,:,::-1])
    '''
    evaluator(out, batch,opt_output=opt_out)
    if i % args.log_freq == args.log_freq - 1:
        evaluator.log()
    #import ipdb
    #ipdb.set_trace()
