import numpy as np
from typing import Dict
import joblib
import torch
from lib.data_utils.img_utils import split_into_chunks
class MoCapDataset:

    def __init__(self, dataset_file: str):
        """
        Dataset class used for loading a dataset of unpaired SMPL parameter annotations
        Args:
            cfg (CfgNode): Model config file.
            dataset_file (str): Path to npz file containing dataset info.
        """
        data = np.load(dataset_file)
        self.pose = data['body_pose'].astype(np.float32)[:, 3:]
        self.betas = data['betas'].astype(np.float32)
        self.length = len(self.pose)
        #import ipdb
        #ipdb.set_trace()

    def __getitem__(self, idx: int) -> Dict:
        pose = self.pose[idx].copy()
        betas = self.betas[idx].copy()
        item = {'body_pose': pose, 'betas': betas}
        return item

    def __len__(self) -> int:
        return self.length

class MoCapDatasetVIBE:

    def __init__(self):
        self.seqlen = 16

        self.stride = 16

        self.db = self.load_db()
        self.vid_indices = split_into_chunks(self.db['vid_name'], self.seqlen, self.stride)
        del self.db['vid_name']
        print(f'AMASS dataset number of videos: {len(self.vid_indices)}')

    def __len__(self):
        return len(self.vid_indices)

    def __getitem__(self, index):
        return self.get_single_item(index)

    def load_db(self):
        db = joblib.load("./data/datasets/amass_db.pt")
        return db

    def get_single_item(self, index):
        start_index, end_index = self.vid_indices[index]
        thetas = self.db['theta'][start_index:end_index+1]

        cam = np.array([1., 0., 0.])[None, ...]
        cam = np.repeat(cam, thetas.shape[0], axis=0)
        theta = np.concatenate([cam, thetas], axis=-1)

        target = {
            'theta': torch.from_numpy(theta).float(),  # cam, pose and shape
        }
        return target