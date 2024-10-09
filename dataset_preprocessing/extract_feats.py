"""
Script used for evaluating the 3D pose errors of TAPE (mode + minimum).

Example usage:
python eval_regression.py --checkpoint=/path/to/checkpoint --dataset=3DPW-TEST

Running the above will compute the Reconstruction Error for the mode as well as the minimum error for the test set of 3DPW.
"""
import torch
import argparse
from tqdm import tqdm
from tape.configs import get_config, tape_config, dataset_config
from tape.models import TAPE
from tape.utils import Evaluator, recursive_to
from tape.datasets import create_dataset
from lib.dataset.loaders import get_data_loaders
from tape.utils.renderer import Renderer

import numpy as np
import os
import cv2
parser = argparse.ArgumentParser(description='Evaluate trained models')
parser.add_argument('--checkpoint', type=str, default='data/_checkpoint.pt', help='Path to pretrained model checkpoint')
parser.add_argument('--model_cfg', type=str, default=None, help='Path to config file. If not set use the default (tape/configs/tape.yaml)')
parser.add_argument('--dataset', type=str, default='H36M-VAL-P2', choices=['H36M-TRAIN','H36M-VAL','H36M-VAL-P2', 'H36M-VAL-P2-OPENPOSE', '3DPW-TEST','3DPW-TRAIN','3DPW-VAL','MPI-INF-TRAIN','MPI-INF_TEST'], help='Dataset to evaluate')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size for inference')
parser.add_argument('--num_workers', type=int, default=0, help='Number of workers used for data loading')
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
model_cfg.TRAIN.NUM_TEST_SAMPLES = 1
model_cfg.freeze()
#checkpoint_spin=torch.load( 'data/spin_model_checkpoint.pth.tar')
#model.load_state_dict(checkpoint_spin,strict=False) 

# Setup model

model = TAPE.load_from_checkpoint(args.checkpoint, strict=False, cfg=model_cfg).to(device)
model.eval()

# Create dataset and data loader
dataset = create_dataset(model_cfg, dataset_cfg, train=False)
dataloader = torch.utils.data.DataLoader(dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)

feats=[]
for i, batch in enumerate(tqdm(dataloader)):
    batch = recursive_to(batch, device)
    with torch.no_grad():
        out = model(batch)
    import ipdb
    ipdb.set_trace()
    for k in range(out['conditioning_feats'].shape[0]):
        feats.append(out['conditioning_feats'][k].cpu().detach().numpy())
np.savez(args.feats_file_name,features=feats)