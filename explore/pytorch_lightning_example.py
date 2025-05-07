"""
General pytorch lightning train pipeline:

for epoch in max_epoch:
    on_train_epoch_start():
        callbacks.on_train_epoch_start(trainer, model)
        model.on_train_epoch_start()
    
    train_outs = []
    for batch_idx, batch in model.train_dataloader():
        on_train_batch_start():    
            callbacks.on_train_batch_start(trainer, model, batch, batch_idx)
            model.on_train_batch_start(batch, batch_idx)
        out = training_step()
        train_outs.append(out)
        loss = out['loss']
        
        on_before_backward():
            callbacks.on_before_backward(trainer, model, loss)
            model.on_before_backward(loss)
        backward():
            loss.backward() 
        on_after_backward():
            callbacks.on_after_backward(trainer, model)
            model.on_after_backward()
        optimizer_step()
        on_before_zero_grad()
        optimizer_zero_grad()

        on_train_batch_end():
            callbacks.on_train_batch_end(trainer, model, train_outs, batch, batch_idx)
            model.on_train_batch_end(train_outs, batch, batch_idx)


    on_validation_epoch_start():
        callbacks.on_validation_epoch_start(trainer, model)
        model.on_validation_epoch_start()
    val_outs = []
    for batch, batch_idx in model.val_dataloader():
        on_validation_batch_start()
            callbacks.on_validation_batch_start(trainer, model, batch, batch_idx)
            model.on_validation_batch_start(batch, batch_idx)
        out = validation_step()
        val_outs.append(out)
        on_validatation_batch_end()
            callbacks.on_validatation_batch_end(trainer, model, val_outs, batch, batch_idx)
            model.on_validatation_batch_end(val_outs, batch, batch_idx)
            
    on_validation_epoch_end()
        callbacks.on_validation_epoch_end(trainer, model)
        model.on_validation_epoch_end()
    
    on_train_epoch_end()
        callbacks.on_train_epoch_end(trainer, model)
        model.on_train_epoch_end()
"""




import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.models.segmentation import fcn_resnet50
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, Callback
from lightning.pytorch.utilities import rank_zero_only
import numpy as np
from PIL import Image
import torchmetrics

from typing_extensions import override

class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir=None, transform=None, target_transform=None, samples=1000):
        """
        A simple segmentation dataset that loads images and masks from directories.
        
        Args:
            root_dir (str): Root directory containing 'images' and 'masks' subdirectories
            transform (callable, optional): Transform to be applied on images
            target_transform (callable, optional): Transform to be applied on masks
        """
        self.image_files = [
            torch.rand(3, 512, 512) for i in range(samples)
        ]
        self.masks = [
            torch.ones(512,512).long() for i in range(samples)
        ]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        return self.image_files[idx], self.masks[idx]

class SegmentationDataModule(pl.LightningDataModule):
    def __init__(self, data_dir=None, batch_size=8, num_workers=4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.target_transform = transforms.Compose([
            transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])
    
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = SegmentationDataset(samples=1000)
            self.val_dataset = SegmentationDataset(samples=200)
            self.test_dataset = SegmentationDataset(samples=500)
        
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,  # Don't shuffle as DistributedSampler will handle it
            pin_memory=True,
            sampler=torch.utils.data.distributed.DistributedSampler(self.train_dataset)
        )
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            sampler=torch.utils.data.distributed.DistributedSampler(self.val_dataset, shuffle=False)
        )
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            sampler=torch.utils.data.distributed.DistributedSampler(self.test_dataset, shuffle=False)
        )

class SegmentationModel(pl.LightningModule):
    def __init__(self, num_classes=21, lr=0.001):
        super().__init__()
        self.model = fcn_resnet50(pretrained=True)
        self.model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        self.lr = lr
        
        # Initialize metrics
        self.train_iou = torchmetrics.JaccardIndex(task="multiclass", num_classes=num_classes)
        self.val_iou = torchmetrics.JaccardIndex(task="multiclass", num_classes=num_classes)
        self.test_iou = torchmetrics.JaccardIndex(task="multiclass", num_classes=num_classes)
        
        # Save hyperparameters for logging
        self.save_hyperparameters()
    
    def forward(self, x):
        outputs = self.model(x)
        return outputs['out']
    
    def on_train_batch_start(self, batch, batch_idx):
        # print("model.on_train_batch_start")
        pass

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # print("model.on_train_batch_end")
        pass

    def on_after_backward(self):
        # print("model.on_after_backward")
        pass
        
    
    def on_before_backward(self, loss):
        # print("model.on_before_backward:", loss)
        pass
        
       
    def training_step(self, batch, batch_idx):
        images, masks = batch
        # Squeeze masks to remove channel dimension if needed
        if masks.dim() == 4 and masks.size(1) == 1:
            masks = masks.squeeze(1)
        
        outputs = self(images)
        loss = F.cross_entropy(outputs, masks.long())
        
        # Calculate IoU
        # preds = torch.argmax(outputs, dim=1)
        # iou = self.train_iou(preds, masks)
        
        # self.log('train_loss', loss, on_step=True, on_epoch=True, sync_dist=True)
        # self.log('train_iou', iou, on_step=False, on_epoch=True, sync_dist=True)
        print("train step loss:", loss)
        return {'loss':loss, 'others':None}
    
    def on_validation_epoch_start(self):
        # print("model.on_validation_epoch_start")
        pass

    def on_validation_batch_start(self, batch, batch_idx):
        # print("model.on_validation_batch_start")
        pass

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        # Squeeze masks to remove channel dimension if needed
        if masks.dim() == 4 and masks.size(1) == 1:
            masks = masks.squeeze(1)
        
        outputs = self(images)
        loss = F.cross_entropy(outputs, masks.long())
        
        # Calculate IoU
        preds = torch.argmax(outputs, dim=1)
        iou = self.val_iou(preds, masks)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_iou', iou, on_step=False, on_epoch=True, sync_dist=True)
        
        return {'val_loss': loss, 'val_iou': iou}
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.1, 
            patience=5,
            verbose=True
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }

class MeanIOUCallback(Callback):
    """
    Custom callback to track mean IoU metric during training 
    and print it at the end of each epoch
    """
    def __init__(self):
        super().__init__()
        
    @rank_zero_only 
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        #print("MeanIOUCallback.on_train_batch_start")
        pass

    @rank_zero_only
    def on_validation_epoch_start(self, trainer, pl_module):
        print("MeanIOUCallback.on_validation_epoch_start")


    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        print("MeanIOUCallback.on_validation_epoch_end")
        # Get the mean IoU value from metrics
        iou_value = trainer.callback_metrics.get('val_iou')
        if iou_value is not None:
            print(f"\nEpoch {trainer.current_epoch}: Validation Mean IoU = {iou_value:.4f}")

class MaxScoreSaver(Callback):
    """
    Custom callback to save model when a metric (like IoU) reaches a new maximum
    Monitors validation IoU by default, but can also track test metrics
    """
    def __init__(self, monitor='val_iou', test_monitor='test_iou', mode='max', save_path='best_models'):
        super().__init__()
        self.monitor = monitor
        self.test_monitor = test_monitor
        self.mode = mode
        self.save_path = save_path
        self.best_score = float('-inf') if mode == 'max' else float('inf')
        self.best_test_score = float('-inf') if mode == 'max' else float('inf')
        # Create directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
    
    @rank_zero_only
    @override 
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        # print("MaxScoreSaver.on_train_batch_start")
        pass
        

    @rank_zero_only 
    @override 
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # print("MaxScoreSaver.on_train_batch_end")
        pass
        

    @rank_zero_only
    @override 
    def on_before_backward(self, trainer, pl_module, loss):
        # print("MaxScoreSaver.on_before_backward:", loss)
        pass
        

    @rank_zero_only
    @override
    def on_after_backward(self, trainer, pl_module):
        # print("MaxScoreSaver.on_after_backward")
        pass
        
        

    @rank_zero_only
    @override
    def on_validation_epoch_start(self, trainer, pl_module):
        # print("MaxScoreSaver.on_validation_epoch_start")
        pass
        

    @override
    def on_validation_epoch_end(self, trainer, pl_module):
        print("MaxScoreSaver.on_validation_epoch_end")
        current_score = trainer.callback_metrics.get(self.monitor)
        
        if current_score is None:
            return
        
        # Convert to scalar if it's a tensor
        if isinstance(current_score, torch.Tensor):
            current_score = current_score.item()
        
        if (self.mode == 'max' and current_score > self.best_score) or \
           (self.mode == 'min' and current_score < self.best_score):
            self.best_score = current_score
            
            # # Only save on rank 0 (main process)
            filepath = os.path.join(self.save_path, f"best_{self.monitor}_{current_score:.4f}.ckpt")
            trainer.save_checkpoint(filepath)
            print(f"\nNew best {self.monitor}: {current_score:.4f}, saved model to {filepath}")

    @rank_zero_only
    @override
    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx):
        # print("MaxScoreSaver.on_validation_batch_start")
        pass

def main():
    # Initialize data module
    data_module = SegmentationDataModule()
    
    # Initialize model
    model = SegmentationModel()
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_iou',
        dirpath='./checkpoints',
        filename='segmentation-{epoch:02d}-{val_iou:.4f}',
        save_top_k=3,
        mode='max',
    )

    mean_iou_callback = MeanIOUCallback()
    max_saver_callback = MaxScoreSaver(monitor='val_iou', test_monitor='test_iou', mode='max')
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=2,
        accelerator='gpu',  # Use GPU for training
        devices=2,  # Number of GPUs
        strategy='ddp_find_unused_parameters_true',  # Distributed Data Parallel
        callbacks=[max_saver_callback],
        precision=32,  # 16 would multiply loss by 65555 to prevent gradient underflow in low precision format
        log_every_n_steps=5000,
        gradient_clip_val=0.5,  # Gradient clipping to prevent exploding gradients
    )
    
    # Train the model
    trainer.fit(model, data_module)
    
    # Final test evaluation (optional as we're already testing after each epoch)
    # print("\nFinal test evaluation:")
    # trainer.test(model, datamodule=data_module)

if __name__ == "__main__":
    main()
