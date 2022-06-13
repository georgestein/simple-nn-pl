import numpy as np
import torch  
import pytorch_lightning as pl

import torch.nn.functional as F

from functools import partial

class MLP(pl.LightningModule):
    """
    From x -> y 
    """

    def __init__(self, hparams):
        super().__init__()
        self.hparams.update(hparams)
        self.save_hyperparameters()    
        
        self.input_dim = self.hparams.get('input_dim', 100)
        self.hidden_dims = self.hparams.get('hidden_dims', [4, 8])
        self.output_dim = self.hparams.get('output_dim', 1)
        self.batch_norm = self.hparams.get('batch_norm', False)

        self.learning_rate = self.hparams.get('learning_rate', 1e-1)
        self.optimizer = self.hparams.get('optimizer', 'Adam')
        self.scheduler = self.hparams.get('scheduler', 'CosineAnnealingLR')
        self.T_max = self.hparams.get('T_max', 25)

        self.scheduler_params = self.hparams.get(
            'scheduler_params',
            {
                'T_max': self.T_max,
                'eta_min': 0,
             }
        )        
        
        mlp = []
        mlp.append(torch.nn.Linear(self.input_dim, self.hidden_dims[0]))
        mlp.append(torch.nn.ReLU())
        if self.batch_norm:
            mlp.append(torch.nn.BatchNorm1d(self.hidden_dims[0]))
                       
        for i in range(len(self.hidden_dims) - 1):
            mlp.append(torch.nn.Linear(self.hidden_dims[i], self.hidden_dims[i+1]))
            mlp.append(torch.nn.ReLU())
            if self.batch_norm:
                mlp.append(torch.nn.BatchNorm1d(self.hidden_dims[i+1]))
                 
        mlp.append(torch.nn.Linear(self.hidden_dims[-1], self.output_dim))
        
        self.mlp = torch.nn.Sequential(*mlp)

    def forward(self, x):
        y = self.mlp(x)
        return y
                   
    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "test")

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x = self._prepare_batch(batch)
        return self(x)

    def configure_optimizers(self):

        optimizer = getattr(torch.optim, self.optimizer)
        optimizer = optimizer(self.parameters(), self.learning_rate)
        
        scheduler = partial(getattr(torch.optim.lr_scheduler, self.scheduler), optimizer)
        scheduler = scheduler(**self.scheduler_params)

        return [optimizer], [scheduler]

    def _prepare_batch(self, batch):
        x, y, = batch
        
        return x, y
        
        
    def _common_step(self, batch, batch_idx, stage: str):
        x, y, = self._prepare_batch(batch)

        loss = F.mse_loss(y, self(x))

        self.log(f"{stage}_loss", loss, on_step=True)

        return loss
