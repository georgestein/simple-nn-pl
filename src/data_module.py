import numpy as np
import torch  
import h5py
import pytorch_lightning as pl

from typing import Any, Callable, Optional, List
import os

class Dataset(torch.utils.data.Dataset):
    """
    Dataset class 

    Data is a .h5 file

    Parameters:
    -----------
    data_path:
        Path to data file
    transform:
        data augmentations to use
    """
    def __init__(
        self,
        data_path,
        transform=None,
    ):
        self.data_path  = data_path
        self.transforms = transform

        with h5py.File(self.data_path, 'r') as hf:
            if all(i in hf.keys() for i in ['x_min', 'x_max', 'y_min', 'y_max']):
                self.x_min, self.x_max = hf['x_min'][:], hf['x_max'][:]
                self.y_min, self.y_max = hf['y_min'][:], hf['y_max'][:]

            else:
                self.x_min, self.x_max = 0., 1.
                self.y_min, self.y_max = 0., 1.

    def _open_file(self):
        self.hfile = h5py.File(self.data_path,'r')

    def __len__(self):
        with h5py.File(self.data_path, 'r') as hf:
            self.n_samples = hf['x'].shape[0]
                
        return self.n_samples

    def min_max_scale(self, dat, dat_min, dat_max, inv=False):

        if not inv:
            return 2*(dat - dat_min)/ (dat_max - dat_min) - 1

        return (dat + 1)/2 * (dat_max - dat_min) + dat_min

    def scale_x(self, x, inv=False):
        return self.min_max_scale(x, self.x_min, self.x_max, inv=inv)

    def scale_y(self, y, inv=False):
        return self.min_max_scale(y, self.y_min, self.y_max, inv=inv)
            
    def __getitem__(self, idx: int):

        if not hasattr(self, 'hfile'):
            self._open_file()

        x, y = self.hfile['x'][idx], self.hfile['y'][idx]  

        x = np.clip(x, self.x_min, self.x_max)
        y = np.clip(y, self.y_min, self.y_max)

        x, y = self.scale_x(x), self.scale_y(y)

        x, y = np.float32(x), np.float32(y)

        return x, y

class DataModule(pl.LightningDataModule):
    """
    Loads data from a single large hdf5 file,
    """
    
    def __init__(
        self,
        params: dict,
        train_transforms=None,
        val_transforms=None,
        test_transforms=None,
    ) -> None:
        
        super().__init__()
        
        self.params = params
        
        self.train_path = self.params.get("train_path", "./")
        self.val_path = self.params.get("val_path", "./") # Set val to same as train
        self.norm_path = self.params.get("norm_path", "./") # path to file containing normalization measured over training set

        self.num_workers = self.params.get("num_workers", 1)
        self.batch_size = self.params.get("batch_size", 4)
        
        self.shuffle = self.params.get("shuffle", True)
        self.pin_memory = self.params.get("pin_memory", True)
        self.drop_last = self.params.get("drop_last", True) # drop last due to queue_size % batch_size == 0. assert in Moco_v2
                
    def _default_transforms(self) -> Callable:

        # transform = DecalsTransforms(self.params)
        transform = None
        
        return transform    
    
    def prepare_data(self) -> None:

        if not os.path.isfile(self.train_path):
            raise FileNotFoundError(
                """
                Your training datafile cannot be found
                """
            )
             
    def setup(self, stage: Optional[str] = None):
        
        # Assign train/val datasets for use in dataloaders     
        if stage == "fit" or stage is None:
    
            train_transforms = self._default_transforms() if self.train_transforms is None else self.train_transforms
            self.train_dataset = Dataset(
                self.train_path,
                train_transforms,
            )
            

            # Val set not used for now
            val_transforms = self._default_transforms() if self.val_transforms is None else self.val_transforms
            self.val_dataset = Dataset(
                self.val_path,
                val_transforms,
            )

        if stage == "predict" or stage is None:

            predict_transforms = CropTransform(self.params)
            # Predict over all training data
            self.predict_dataset = Dataset(
                self.train_path,
                predict_transforms,
            )

                    
    def train_dataloader(self):
 
        loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )
            
        return loader
    
    def val_dataloader(self):

        loader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )       
        
        return loader

    def predict_dataloader(self):
        
        loader = torch.utils.data.DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )       
        
        return loader
