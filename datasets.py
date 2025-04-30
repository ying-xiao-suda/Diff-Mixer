import pickle
import os
import re
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import h5py
import pickle

def sample_mask(shape, p=0.0015, p_noise=0.05, max_seq=48, min_seq=12, rng=None
                ):
    if rng is None:
        rand = np.random.random
        randint = np.random.randint
    else:
        rand = rng.random
        randint = rng.integers
    mask = rand(shape) < p
    for col in range(mask.shape[1]):
        idxs = np.flatnonzero(mask[:, col])
        if not len(idxs):
            continue
        fault_len = min_seq
        if max_seq > min_seq:
            fault_len = fault_len + int(randint(max_seq - min_seq))
        idxs_ext = np.concatenate([np.arange(i, i + fault_len) for i in idxs])
        idxs = np.unique(idxs_ext)
        idxs = np.clip(idxs, 0, shape[0] - 1)
        mask[idxs, col] = True
    mask = mask | (rand(mask.shape) < p_noise)
    return 1-mask.astype('uint8')


def parse_id(data,missing_ratio=0.1,block=False,rng=None):
    observed_values = np.array(data)

    observed_masks = ~np.isnan(observed_values)

    if(block):
        gt_masks = sample_mask(observed_masks.shape,rng=rng)
    else:    
        # gt_masks = masks.reshape(observed_masks.shape)
        gt_masks = sample_mask(observed_masks.shape, p=0.0, p_noise=missing_ratio, max_seq=1, min_seq=1, rng=rng
                )
    observed_values = np.nan_to_num(observed_values)
    observed_masks = observed_masks.astype("float32")
    gt_masks = gt_masks.astype("float32")
    
    return observed_values, observed_masks, gt_masks



class bay_Dataset(Dataset):
    def __init__(self,config, eval_length=288, missing_ratio=0.0, seed=0,mode=0,block=False):

        with h5py.File('data/bay/pems-bay.h5', 'r') as file:

            df_group = file['speed']

            for dataset_name in df_group:
                dataset = df_group[dataset_name]
       
                data = dataset[:]
                

                df = pd.DataFrame(data)
        data=df.values
        data = np.insert(data, 20184, np.full((12, data.shape[1]), np.nan), axis=0)

        self.seq_len=config['seq_len']
        train_index=np.arange(0,int(0.7*len(data)))
        valid_index=np.arange(int(0.7*len(data)),int(0.8*len(data)))
        test_index=np.arange(int(0.8*len(data)),int(len(data)))
        if mode==1:
            use_index_list=train_index
        elif mode==2:
            use_index_list=valid_index
        elif mode==3:
            use_index_list=test_index
        elif mode==0:
            use_index_list=np.arange(0,181)

        self.mode=mode
        self.eval_length = eval_length
        np.random.seed(seed)  # seed for ground truth choice

        #data shape:(52128, 325)

        self.mean = np.zeros(325)
        self.std = np.zeros(325)
        self.truth=None
        self.block=block
        
        self.rng = np.random.default_rng(seed)

        path = (
            "./data/bay_missing/bay_missing" + str(missing_ratio) +'mode'+str(mode)+ "_seed" + str(seed) +'block_'+str(self.block)+ ".pk"
        )

        if use_index_list is None:
            self.use_index_list = np.arange(len(self.observed_values))
        else:
            self.use_index_list = use_index_list

        if os.path.isfile(path) == False:  # if datasetfile is none, create
            self.observed_values, self.observed_masks, self.gt_masks = parse_id(
                        data, missing_ratio,block=self.block,rng=self.rng
                    )    
 
            self.truth=self.observed_values
    
            self.truth=self.observed_values
            # calc mean and std and normalize values
            # (it is the same normalization as Cao et al. (2018) (https://github.com/caow13/BRITS))

            tmp_values = self.observed_values.reshape(-1, 325)
            tmp_masks = self.observed_masks.reshape(-1,325).astype(int)
            tmp_masks1=self.gt_masks.reshape(-1,325).astype(int)
            tmp_masks1[np.arange((len(train_index)+len(valid_index)))]=1
            tmp_masks=tmp_masks&tmp_masks1
            for k in range(325):
                c_data = tmp_values[:, k][tmp_masks[:, k] == 1]
                self.mean[k] = c_data.mean()
                self.std[k] = c_data.std()

            self.observed_values = (
                (self.observed_values - self.mean) / self.std * self.observed_masks
            )
            self.observed_values=self.observed_values[use_index_list]
            self.observed_masks=self.observed_masks[use_index_list]
            self.gt_masks=self.gt_masks[use_index_list]
            self.truth=self.truth[use_index_list]
            with open(path, "wb") as f:
                pickle.dump(
                    [self.observed_values, self.observed_masks, self.gt_masks,self.mean,self.std,self.truth], f
                )
        else:  # load datasetfile
            with open(path, "rb") as f:
                self.observed_values, self.observed_masks, self.gt_masks,self.mean,self.std,self.truth = pickle.load(
                    f
                )

    def __getitem__(self, index):
        start = index * self.seq_len
        end = start + self.seq_len
        s = {
            "observed_data": self.observed_values[start:end,:],
            "observed_mask": self.observed_masks[start:end,:],
            "gt_mask": self.gt_masks[start:end,:],
            "timepoints": np.arange(self.seq_len), 
        }                      

        return s

    def __len__(self):
        N = len(self.use_index_list)
        if N < self.seq_len:
            return 0
        return (N - self.seq_len) // self.seq_len + 1
    
    def get_mean_std(self):
        return self.mean, self.std
    
    def get_truth(self):
        return self.truth

def get_dataloader_bay(config,seed=1, batch_size=16, missing_ratio=0.1,block=False):

    dataset = bay_Dataset(config,missing_ratio=missing_ratio, seed=seed,mode=1,block=block)
    mean,std=dataset.get_mean_std()
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=1)
    valid_dataset = bay_Dataset(config,missing_ratio=missing_ratio, seed=seed,mode=2,block=block)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=0)
    test_dataset = bay_Dataset(config,missing_ratio=missing_ratio, seed=seed,mode=3,block=block)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=0)
    
    return train_loader, valid_loader, test_loader,mean,std



class Metrla_Dataset(Dataset):
    def __init__(self,config, eval_length=288, missing_ratio=0.0, seed=0,mode=0,block=False):
        with h5py.File('data/metrla/metr-la.h5', 'r') as file:

            df_group = file['/df']
            
    
            for dataset_name in df_group:
                dataset = df_group[dataset_name]
                
    
                data = dataset[:]
                
                
                df = pd.DataFrame(data)
        data=df.values
        self.seq_len=config['seq_len']
 
        train_index=np.arange(0,int(0.7*len(data)))
        valid_index=np.arange(int(0.7*len(data)),int(0.8*len(data)))
        test_index=np.arange(int(0.8*len(data)),int(len(data)))
        if mode==1:
            use_index_list=train_index
        elif mode==2:
            use_index_list=valid_index
        elif mode==3:
            use_index_list=test_index
        elif mode==0:
            use_index_list=np.arange(0,119)

        self.mode=mode
        self.eval_length = eval_length
        np.random.seed(seed)  # seed for ground truth choice
        #data shape:(34272, 207)

        self.mean = np.zeros(207)
        self.std = np.zeros(207)
        self.truth=None
        self.block=block
        self.rng = np.random.default_rng(seed)

        path = (
            "./data/Metrla_missing/Metrla_missing" + str(missing_ratio) +'mode'+str(mode)+ "_seed" + str(seed) +'block_'+str(self.block)+ ".pk"
        )

        if use_index_list is None:
            self.use_index_list = np.arange(len(self.observed_values))
        else:
            self.use_index_list = use_index_list

        if os.path.isfile(path) == False:  # if datasetfile is none, create
            self.observed_values, self.observed_masks, self.gt_masks = parse_id(
                        data, missing_ratio,block=self.block,rng=self.rng
                    )  
         
            self.truth=self.observed_values


            # calc mean and std and normalize values
            # (it is the same normalization as Cao et al. (2018) (https://github.com/caow13/BRITS))
        
            tmp_values = self.observed_values.reshape(-1, 207)
            tmp_masks = self.observed_masks.reshape(-1,207).astype(int)
            tmp_masks1=self.gt_masks.reshape(-1,207).astype(int)
            tmp_masks1[np.arange((len(train_index)+len(valid_index)))]=1
            tmp_masks=tmp_masks&tmp_masks1
            for k in range(207):
                c_data = tmp_values[:, k][tmp_masks[:, k] == 1]
                self.mean[k] = c_data.mean()
                self.std[k] = c_data.std()

            self.observed_values = (
                (self.observed_values - self.mean) / self.std * self.observed_masks
            )
            self.observed_values=self.observed_values[use_index_list]
            self.observed_masks=self.observed_masks[use_index_list]
            self.gt_masks=self.gt_masks[use_index_list]
            self.truth=self.truth[use_index_list]
            with open(path, "wb") as f:
                pickle.dump(
                    [self.observed_values, self.observed_masks, self.gt_masks,self.mean,self.std,self.truth], f
                )
        else:  # load datasetfile
            with open(path, "rb") as f:
                self.observed_values, self.observed_masks, self.gt_masks,self.mean,self.std,self.truth = pickle.load(
                    f
                )

    def __getitem__(self, index):
        start = index * self.seq_len
        end = start + self.seq_len
        s = {
            "observed_data": self.observed_values[start:end,:],
            "observed_mask": self.observed_masks[start:end,:],
            "gt_mask": self.gt_masks[start:end,:],
            "timepoints": np.arange(self.seq_len), 
        }                      

        return s

    def __len__(self):
        N = len(self.use_index_list)
        if N < self.seq_len:
            return 0
        return (N - self.seq_len) // self.seq_len + 1
    
    def get_mean_std(self):
        return self.mean, self.std
    
    def get_truth(self):
        return self.truth


def get_dataloader_Metrla(config,seed=1, batch_size=16, missing_ratio=0.1,block=False):

    dataset = Metrla_Dataset(config,missing_ratio=missing_ratio, seed=seed,mode=1,block=block)
    mean,std=dataset.get_mean_std()
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=1)
    valid_dataset = Metrla_Dataset(config,missing_ratio=missing_ratio, seed=seed,mode=2,block=block)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=0)
    test_dataset = Metrla_Dataset(config,missing_ratio=missing_ratio, seed=seed,mode=3,block=block)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=0)
    
    return train_loader, valid_loader, test_loader,mean,std



class Pems04_Dataset(Dataset):
    def __init__(self,config, eval_length=288, missing_ratio=0.0, seed=0,mode=0,block=False):

        self.seq_len=config['seq_len']
        self.da_len=288//self.seq_len
        data = np.load("./data/pems04/pems04.npz")
        data=data['data'][:,:,0]
        train_index=np.arange(0,int(0.7*len(data)))
        valid_index=np.arange(int(0.7*len(data)),int(0.8*len(data)))
        test_index=np.arange(int(0.8*len(data)),int(len(data)))
        if mode==1:
            use_index_list=train_index
        elif mode==2:
            use_index_list=valid_index
        elif mode==3:
            use_index_list=test_index
        elif mode==0:
            use_index_list=np.arange(0,int(len(data)))

        self.mode=mode
        self.eval_length = eval_length
        np.random.seed(seed)  # seed for ground truth choice
        #data shape:(16992, 307)

        self.mean = np.zeros(307)
        self.std = np.zeros(307)
        self.truth=None
        self.block=block
        self.rng = np.random.default_rng(seed)

        path = (
            "./data/Pems04_missing/Pems04_missing" + str(missing_ratio) +'mode'+str(mode)+ "_seed" + str(seed) +'block_'+str(self.block)+ ".pk"
        )

        if use_index_list is None:
            self.use_index_list = np.arange(len(self.observed_values))
        else:
            self.use_index_list = use_index_list

        if os.path.isfile(path) == False:  # if datasetfile is none, create

            self.observed_values, self.observed_masks, self.gt_masks = parse_id(
                        data, missing_ratio,block=self.block,rng=self.rng
                    )    
 
            self.truth=self.observed_values
                

            self.observed_values = np.array(self.observed_values)
            self.observed_masks = np.array(self.observed_masks)
            self.gt_masks = np.array(self.gt_masks)
       
            self.truth=self.observed_values


            # calc mean and std and normalize values
            # (it is the same normalization as Cao et al. (2018) (https://github.com/caow13/BRITS))
 
            tmp_values = self.observed_values.reshape(-1, 307)
            tmp_masks = self.observed_masks.reshape(-1,307).astype(int)
            tmp_masks1=self.gt_masks.reshape(-1,307).astype(int)
            tmp_masks1[np.arange((len(train_index)+len(valid_index)))]=1
            tmp_masks=tmp_masks&tmp_masks1
            for k in range(307):
                c_data = tmp_values[:, k][tmp_masks[:, k] == 1]
                self.mean[k] = c_data.mean()
                self.std[k] = c_data.std()

            self.observed_values = (
                (self.observed_values - self.mean) / self.std * self.observed_masks
            )
            self.observed_values=self.observed_values[use_index_list]
            self.observed_masks=self.observed_masks[use_index_list]
            self.gt_masks=self.gt_masks[use_index_list]
            self.truth=self.truth[use_index_list]
            with open(path, "wb") as f:
                pickle.dump(
                    [self.observed_values, self.observed_masks, self.gt_masks,self.mean,self.std,self.truth], f
                )
        else:  # load datasetfile
            with open(path, "rb") as f:
                self.observed_values, self.observed_masks, self.gt_masks,self.mean,self.std,self.truth = pickle.load(
                    f
                )

    def __getitem__(self, index):
        start = index * self.seq_len
        end = start + self.seq_len
        s = {
            "observed_data": self.observed_values[start:end,:],
            "observed_mask": self.observed_masks[start:end,:],
            "gt_mask": self.gt_masks[start:end,:],
            "timepoints": np.arange(self.seq_len), 
        }                      

        return s

    def __len__(self):
        N = len(self.use_index_list)
        if N < self.seq_len:
            return 0
        return (N - self.seq_len) // self.seq_len + 1
    
    def get_mean_std(self):
        return self.mean, self.std
    
    def get_truth(self):
        return self.truth


def get_dataloader_Pems04(config,seed=1,batch_size=16, missing_ratio=0.1,block=False):

    dataset = Pems04_Dataset(config,missing_ratio=missing_ratio, seed=seed,mode=1,block=block)
    mean,std=dataset.get_mean_std()
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=1)
    valid_dataset = Pems04_Dataset(config,missing_ratio=missing_ratio, seed=seed,mode=2,block=block)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=0)
    test_dataset = Pems04_Dataset(config,missing_ratio=missing_ratio, seed=seed,mode=3,block=block)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=0)
    
    return train_loader, valid_loader, test_loader,mean,std


class Pems08_Dataset(Dataset):
    def __init__(self,config, eval_length=288, missing_ratio=0.0, seed=0,mode=0,block=False):
        
        data = np.load("./data/pems08/pems08.npz")
        data=data['data'][:,:,0]
        self.seq_len=config['seq_len']
        train_index=np.arange(0,int(0.7*len(data)))
        valid_index=np.arange(int(0.7*len(data)),int(0.8*len(data)))
        test_index=np.arange(int(0.8*len(data)),int(len(data)))
        if mode==1:
            use_index_list=train_index
        elif mode==2:
            use_index_list=valid_index
        elif mode==3:
            use_index_list=test_index
        elif mode==0:
            use_index_list=np.arange(0,len(data))

        self.mode=mode
        self.eval_length = eval_length
        np.random.seed(seed)  # seed for ground truth choice
        #data shape:(16992, 307)
        self.mean = np.zeros(170)
        self.std = np.zeros(170)
        self.truth=None
        self.block=block
        self.rng = np.random.default_rng(seed)

        path = (
            "./data/Pems08_missing/Pems08_missing" + str(missing_ratio) +'mode'+str(mode)+ "_seed" + str(seed) +'block_'+str(self.block)+ ".pk"
        )

        if use_index_list is None:
            self.use_index_list = np.arange(len(self.observed_values))
        else:
            self.use_index_list = use_index_list

        if os.path.isfile(path) == False:  # if datasetfile is none, create

            self.observed_values, self.observed_masks, self.gt_masks = parse_id(
                        data, missing_ratio,block=self.block,rng=self.rng
                    )    
 
            self.truth=self.observed_values

            # calc mean and std and normalize values
            # (it is the same normalization as Cao et al. (2018) (https://github.com/caow13/BRITS))
            
            tmp_values = self.observed_values.reshape(-1, 170)
            tmp_masks = self.observed_masks.reshape(-1,170).astype(int)
            tmp_masks1=self.gt_masks.reshape(-1,170).astype(int)
            tmp_masks1[np.arange((len(train_index)+len(valid_index)))]=1
            tmp_masks=tmp_masks&tmp_masks1
            for k in range(170):
                c_data = tmp_values[:, k][tmp_masks[:, k] == 1]
                self.mean[k] = c_data.mean()
                self.std[k] = c_data.std()

            self.observed_values = (
                (self.observed_values - self.mean) / self.std * self.observed_masks
            )
            self.observed_values=self.observed_values[use_index_list]
            self.observed_masks=self.observed_masks[use_index_list]
            self.gt_masks=self.gt_masks[use_index_list]
            self.truth=self.truth[use_index_list]
            with open(path, "wb") as f:
                pickle.dump(
                    [self.observed_values, self.observed_masks, self.gt_masks,self.mean,self.std,self.truth], f
                )
        else:  # load datasetfile
            with open(path, "rb") as f:
                self.observed_values, self.observed_masks, self.gt_masks,self.mean,self.std,self.truth = pickle.load(
                    f
                )

    def __getitem__(self, index):
        start = index * self.seq_len
        end = start + self.seq_len
        s = {
            "observed_data": self.observed_values[start:end,:],
            "observed_mask": self.observed_masks[start:end,:],
            "gt_mask": self.gt_masks[start:end,:],
            "timepoints": np.arange(self.seq_len), 
        }                      

        return s

    def __len__(self):
        N = len(self.use_index_list)
        if N < self.seq_len:
            return 0
        return (N - self.seq_len) // self.seq_len + 1
    
    def get_mean_std(self):
        return self.mean, self.std
    
    def get_truth(self):
        return self.truth



def get_dataloader_Pems08(config,seed=1, batch_size=16, missing_ratio=0.1,block=False):

    dataset = Pems08_Dataset(config,missing_ratio=missing_ratio, seed=seed,mode=1,block=block)
    mean,std=dataset.get_mean_std()
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=1)
    valid_dataset = Pems08_Dataset(config,missing_ratio=missing_ratio, seed=seed,mode=2,block=block)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=0)
    test_dataset = Pems08_Dataset(config,missing_ratio=missing_ratio, seed=seed,mode=3,block=block)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=0)
    
    return train_loader, valid_loader, test_loader,mean,std


class Pems07_Dataset(Dataset):
    def __init__(self,config, eval_length=288, missing_ratio=0.0, seed=0,mode=0,block=False):
       
        self.seq_len=config['seq_len']
        self.da_len=288//self.seq_len
        data =np.load("./data/pems07/pems07.npz")
        data=data['data']
        data = data.reshape(28224, 883)
        train_index=np.arange(0,int(0.7*len(data)))
        valid_index=np.arange(int(0.7*len(data)),int(0.8*len(data)))
        test_index=np.arange(int(0.8*len(data)),int(len(data)))
        if mode==1:
            use_index_list=train_index
        elif mode==2:
            use_index_list=valid_index
        elif mode==3:
            use_index_list=test_index
        elif mode==0:
            use_index_list=np.arange(0,98)

        self.mode=mode
        self.eval_length = eval_length
        np.random.seed(seed)  # seed for ground truth choice

        self.observed_values = []
        self.observed_masks = []
        self.gt_masks = []
        self.mean = np.zeros(883)
        self.std = np.zeros(883)
        self.truth=None
        self.block=block
        self.rng = np.random.default_rng(seed)

        path = (
            "./data/Pems07_missing/Pems07_missing" + str(missing_ratio) +'mode'+str(mode)+ "_seed" + str(seed) +'block_'+str(self.block)+ ".pk"
        )

        if use_index_list is None:
            self.use_index_list = np.arange(len(self.observed_values))
        else:
            self.use_index_list = use_index_list

        if os.path.isfile(path) == False:  # if datasetfile is none, create
            self.observed_values, self.observed_masks, self.gt_masks = parse_id(
                        data, missing_ratio,block=self.block,rng=self.rng
                    )    
 
            self.truth=self.observed_values
                

            self.truth=self.observed_values


            # calc mean and std and normalize values
            # (it is the same normalization as Cao et al. (2018) (https://github.com/caow13/BRITS))
           
            tmp_values = self.observed_values.reshape(-1, 883)
            tmp_masks = self.observed_masks.reshape(-1,883).astype(int)
            tmp_masks1=self.gt_masks.reshape(-1,883).astype(int)
            tmp_masks1[np.arange((len(train_index)+len(valid_index)))]=1
            tmp_masks=tmp_masks&tmp_masks1
            for k in range(883):
                c_data = tmp_values[:, k][tmp_masks[:, k] == 1]
                self.mean[k] = c_data.mean()
                self.std[k] = c_data.std()
            self.observed_values = (
                (self.observed_values - self.mean) / self.std * self.observed_masks
            )
            self.observed_values=self.observed_values[use_index_list]
            self.observed_masks=self.observed_masks[use_index_list]
            self.gt_masks=self.gt_masks[use_index_list]
            self.truth=self.truth[use_index_list]
            with open(path, "wb") as f:
                pickle.dump(
                    [self.observed_values, self.observed_masks, self.gt_masks,self.mean,self.std,self.truth], f
                )
        else:  # load datasetfile
            with open(path, "rb") as f:
                self.observed_values, self.observed_masks, self.gt_masks,self.mean,self.std,self.truth = pickle.load(
                    f
                )

    def __getitem__(self, index):
        start = index * self.seq_len
        end = start + self.seq_len
        s = {
            "observed_data": self.observed_values[start:end,:],
            "observed_mask": self.observed_masks[start:end,:],
            "gt_mask": self.gt_masks[start:end,:],
            "timepoints": np.arange(self.seq_len), 
        }                      

        return s

    def __len__(self):
        N = len(self.use_index_list)
        if N < self.seq_len:
            return 0
        return (N - self.seq_len) // self.seq_len + 1
    
    def get_mean_std(self):
        return self.mean, self.std
    
    def get_truth(self):
        return self.truth

def get_dataloader_Pems07(config,seed=1, batch_size=16, missing_ratio=0.1,block=False):

    dataset = Pems07_Dataset(config,missing_ratio=missing_ratio, seed=seed,mode=1,block=block)
    mean,std=dataset.get_mean_std()
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=1)
    valid_dataset = Pems07_Dataset(config,missing_ratio=missing_ratio, seed=seed,mode=2,block=block)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=0)
    test_dataset = Pems07_Dataset(config,missing_ratio=missing_ratio, seed=seed,mode=3,block=block)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=0)
    
    return train_loader, valid_loader, test_loader,mean,std


class Pems03_Dataset(Dataset):
    def __init__(self,config, eval_length=288, missing_ratio=0.0, seed=0,mode=0,block=False):
    
        self.seq_len=config['seq_len']
        data =np.load("./data/pems03/pems03.npz")
        data=data['data']
        data = data.reshape(26208, 358)

        train_index=np.arange(0,int(0.7*len(data)))
        valid_index=np.arange(int(0.7*len(data)),int(0.8*len(data)))
        test_index=np.arange(int(0.8*len(data)),int(len(data)))
        if mode==1:
            use_index_list=train_index
        elif mode==2:
            use_index_list=valid_index
        elif mode==3:
            use_index_list=test_index
        elif mode==0:
            use_index_list=np.arange(0,91)

        self.mode=mode
        self.eval_length = eval_length
        np.random.seed(seed)  # seed for ground truth choice


        self.mean = np.zeros(358)
        self.std = np.zeros(358)
        self.truth=None
        self.block=block
        self.rng = np.random.default_rng(seed)

        path = (
            "./data/Pems03_missing/Pems03_missing" + str(missing_ratio) +'mode'+str(mode)+ "_seed" + str(seed) +'block_'+str(self.block)+ ".pk"
        )

        if use_index_list is None:
            self.use_index_list = np.arange(len(self.observed_values))
        else:
            self.use_index_list = use_index_list

        if os.path.isfile(path) == False:  # if datasetfile is none, create
            self.observed_values, self.observed_masks, self.gt_masks = parse_id(
                        data, missing_ratio,block=self.block,rng=self.rng
                    )    
 
            self.truth=self.observed_values
                
 
            self.truth=self.observed_values


            # calc mean and std and normalize values
            # (it is the same normalization as Cao et al. (2018) (https://github.com/caow13/BRITS))
  
            tmp_values = self.observed_values.reshape(-1, 358)
            tmp_masks = self.observed_masks.reshape(-1,358).astype(int)
            tmp_masks1=self.gt_masks.reshape(-1,358).astype(int)
            tmp_masks1[np.arange((len(train_index)+len(valid_index)))]=1
            tmp_masks=tmp_masks&tmp_masks1
            for k in range(358):
                c_data = tmp_values[:, k][tmp_masks[:, k] == 1]
                self.mean[k] = c_data.mean()
                self.std[k] = c_data.std()
            self.observed_values = (
                (self.observed_values - self.mean) / self.std * self.observed_masks
            )
            self.observed_values=self.observed_values[use_index_list]
            self.observed_masks=self.observed_masks[use_index_list]
            self.gt_masks=self.gt_masks[use_index_list]
            self.truth=self.truth[use_index_list]
            with open(path, "wb") as f:
                pickle.dump(
                    [self.observed_values, self.observed_masks, self.gt_masks,self.mean,self.std,self.truth], f
                )
        else:  # load datasetfile
            with open(path, "rb") as f:
                self.observed_values, self.observed_masks, self.gt_masks,self.mean,self.std,self.truth = pickle.load(
                    f
                )

    def __getitem__(self, index):
        start = index * self.seq_len
        end = start + self.seq_len
        s = {
            "observed_data": self.observed_values[start:end,:],
            "observed_mask": self.observed_masks[start:end,:],
            "gt_mask": self.gt_masks[start:end,:],
            "timepoints": np.arange(self.seq_len), 
        }                      

        return s

    def __len__(self):
        N = len(self.use_index_list)
        if N < self.seq_len:
            return 0
        return (N - self.seq_len) // self.seq_len + 1
    
    def get_mean_std(self):
        return self.mean, self.std
    
    def get_truth(self):
        return self.truth

def get_dataloader_Pems03(config,seed=1, batch_size=16, missing_ratio=0.1,block=False):

    dataset = Pems03_Dataset(config,missing_ratio=missing_ratio, seed=seed,mode=1,block=block)
    mean,std=dataset.get_mean_std()
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=1)
    valid_dataset = Pems03_Dataset(config,missing_ratio=missing_ratio, seed=seed,mode=2,block=block)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=0)
    test_dataset = Pems03_Dataset(config,missing_ratio=missing_ratio, seed=seed,mode=3,block=block)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=0)
    
    return train_loader, valid_loader, test_loader,mean,std

