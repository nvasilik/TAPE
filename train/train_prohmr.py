"""
Script used to train TAPE.
Example usage:
python train_tape.py --root_dir=/path/to/experiment/folder

Running the above will use the default config file to train TAPE as in the paper.
The code uses PyTorch Lightning for training.
"""
import os
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from tape.configs import get_config, tape_config, dataset_config
from tape.datasets import TAPEDataModule
from tape.models import TAPE

parser = argparse.ArgumentParser(description='TAPE training code')
parser.add_argument('--model_cfg', type=str, default=None, help='Path to config file')
parser.add_argument('--root_dir', type=str, required=True, help='Directory to save logs and checkpoints')


args = parser.parse_args()

# Load model config
if args.model_cfg is None:
    model_cfg = tape_config()
else:
    model_cfg = get_config(args.model_cfg)

# Load dataset config
dataset_cfg = dataset_config()

# Setup training and validation datasets
data_module = TAPEDataModule(model_cfg, dataset_cfg)
# Setup model
model = TAPE(model_cfg)

# Setup Tensorboard logger
logger = TensorBoardLogger(os.path.join(args.root_dir, 'tensorboard'), name='', version='', default_hp_metric=False)

# Setup checkpoint saving
checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=os.path.join(args.root_dir, 'checkpoints'), every_n_train_steps=model_cfg.GENERAL.CHECKPOINT_STEPS)
checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=os.path.join(args.root_dir, 'checkpoints'), monitor='val_loss',save_top_k=6,mode='min',save_last=True)
#model
#import ipdb
#ipdb.set_trace()
#checkpoint = torch.load( 'tape_reproduce/checkpoints/provibe_3dpw_10k_eval.ckpt')

#model.load_state_dict(checkpoint['state_dict'],strict=False) 
#checkpoint_vibe=torch.load( 'data/vibe_model_w_3dpw.pth.tar')
#checkpoint_vibe=torch.load('data/model_best.pth.tar')
#temp = {"weight_ih_l0":checkpoint_vibe['gen_state_dict']['encoder.gru.weight_ih_l0'],
#        "weight_hh_l0":checkpoint_vibe['gen_state_dict']['encoder.gru.weight_hh_l0'],
#        "bias_ih_l0":checkpoint_vibe['gen_state_dict']['encoder.gru.bias_ih_l0'],
#        "bias_hh_l0":checkpoint_vibe['gen_state_dict']['encoder.gru.bias_hh_l0'],
#        "weight":checkpoint_vibe['gen_state_dict']['encoder.linear.weight'],
#        "bias":checkpoint_vibe['gen_state_dict']['encoder.linear.bias']
#    
#    }
#import ipdb
#ipdb.set_trace()
#model.encoder.gru.load_state_dict(temp,strict=False)

#checkpoint_disc=torch.load('data/model_best.pth.tar')
#checkpoint_spin=torch.load('data/spin_model_checkpoint.pth.tar')

#model.backbone.load_state_dict(checkpoint_spin['model'],strict=False)
checkpoint = torch.load('data/_checkpoint.pt')
#checkpoint = torch.load('tape_reproduce/checkpoints/last.ckpt')

model.load_state_dict(checkpoint['state_dict'],strict=False)


#, 'encoder.gru.weight_hh_l0', 'encoder.gru.bias_ih_l0', 'encoder.gru.bias_hh_l0'
#dmodel.backbone.requires_grad=False
# Setup PyTorch Lightning Trainer
for param in model.backbone.parameters():
    param.requires_grad=False
#if model_cfg.MODE.MOTION_DISCRIMINATOR:
#    for param in model.motion_discriminator.parameters():
#        param.requires_grad=False
#pl.seed_everything(12345,workers=True)
trainer = pl.Trainer(default_root_dir=args.root_dir,
                     #deterministic=True,#check
                     logger=logger,
                     gpus=1,
                     limit_val_batches=1,
                     num_sanity_val_steps=0,
                     log_every_n_steps=model_cfg.GENERAL.LOG_STEPS,
                     flush_logs_every_n_steps=model_cfg.GENERAL.LOG_STEPS,
                     val_check_interval=model_cfg.GENERAL.VAL_STEPS,
                     progress_bar_refresh_rate=1,
                     precision=16,
                     max_steps=model_cfg.GENERAL.TOTAL_STEPS,
                     move_metrics_to_cpu=True,
                     callbacks=[checkpoint_callback])
                     #resume_from_checkpoint='tape_reproduce/checkpoints/last.ckpt')
#model = model.load_from_checkpoint("data/_checkpoint.pt")
#checkpointtcmr=torch.load('tcmr_table4_3dpw_test.pth.tar')
#checkpoint = torch.load('data/_checkpoint.pt')
#model.load_state_dict(checkpointtcmr['gen_state_dict'],strict=False)
#model.load_state_dict(checkpoint['state_dict'],strict=False)
#if model_cfg.MODE.MOTION_DISCRIMINATOR:
#    model.motion_discriminator.load_state_dict(checkpoint_disc['disc_motion_state_dict'],strict=False)

#model.optimizers.load_state_dict(checkpoint['optimizer_states'],strict=False)
#model.lr_schedulers.load_state_dict(checkpoint['lr_schedulers'],strict=False)
#import ipdb
#ipdb.set_trace()
#model.load_state_dict(checkpoint)
# Train the model

trainer.fit(model, datamodule=data_module)
