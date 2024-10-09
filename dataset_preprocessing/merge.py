import numpy as np

file_list=['mpi_train_feats.npz','mpi_inf_3dhp_train.npz']
data_all = [np.load(fname,allow_pickle=True) for fname in file_list]
merged_data = {}
for data in data_all:
    [merged_data.update({k: v}) for k, v in data.items()]
np.savez('new_file.npz', **merged_data)
