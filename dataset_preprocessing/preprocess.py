"""
Script used for evaluating the 3D pose errors of TAPE (mode + minimum).

Example usage:
python eval_regression.py --checkpoint=/path/to/checkpoint --dataset=3DPW-TEST

Running the above will compute the Reconstruction Error for the mode as well as the minimum error for the test set of 3DPW.
"""
import ipdb
import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from tape.configs import get_config, tape_config, dataset_config
from tape.models import TAPE
from tape.utils import Evaluator, recursive_to
from tape.datasets import create_dataset

parser = argparse.ArgumentParser(description='Evaluate trained models')
parser.add_argument('--checkpoint', type=str, default='data/checkpoint.pt', help='Path to pretrained model checkpoint')
parser.add_argument('--model_cfg', type=str, default=None, help='Path to config file. If not set use the default (tape/configs/tape.yaml)')
parser.add_argument('--dataset', type=str, default='H36M-VAL-P2', choices=['H36M-TRAIN','H36M-VAL-P2', '3DPW-TEST', 'MPI-INF-TEST'], help='Dataset to evaluate')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for inference')
parser.add_argument('--num_samples', type=int, default=1, help='Number of test samples to draw')
parser.add_argument('--num_workers', type=int, default=4, help='Number of workers used for data loading')
parser.add_argument('--log_freq', type=int, default=10, help='How often to log results')
parser.add_argument('--shuffle', dest='shuffle', action='store_true', default=False, help='Shuffle the dataset during evaluation')


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
model = TAPE.load_from_checkpoint(args.checkpoint, strict=False, cfg=model_cfg).to(device)
model.eval()

# Create dataset and data loader
dataset = create_dataset(model_cfg, dataset_cfg, train=False)
dataloader = torch.utils.data.DataLoader(dataset, args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers)

# List of metrics to log
metrics = ['mode_re', 'min_re']

# Setup evaluator object
evaluator = Evaluator(dataset_length=len(dataset), keypoint_list=dataset_cfg.KEYPOINT_LIST, pelvis_ind=model_cfg.EXTRA.PELVIS_IND, metrics=metrics)

root_dir = '/media/nikos/data/h36m/processed_tape'

# Go over the images in the dataset.
for i, batch in enumerate(tqdm(dataloader)):
    batch = recursive_to(batch, device)
    with torch.no_grad():
        out = model(batch)
    feats = out['conditioning_feats']
    for n,f in enumerate(feats):
        imgname = batch['imgname'][n]
        path = batch['imgname'][n]
        path = path.split('/')[-1].replace('.jpg','.npy')
        subject = path.split('_')[0]
        seq = path.split('.')[0][len(subject)+1:]
        camera = path.split('.')[1].split('_')[0]
        frame = path.split('.')[1].split('_')[1]
        out_dir = os.path.join(root_dir,subject,seq,camera)
        os.makedirs(out_dir,exist_ok=True)
        path = os.path.join(out_dir,frame+'.npy')
        np.save(path,f.cpu().numpy())
    evaluator(out, batch)
    if i % args.log_freq == args.log_freq - 1:
        evaluator.log()