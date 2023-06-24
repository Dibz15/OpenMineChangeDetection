import lightning as pl
import torch
from .change_classifier import ChangeClassifier

class ChangeDetectorLightningModule(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.change_detector = ChangeClassifier(**kwargs)
        self.criterion = torch.nn.BCELoss()
        self.save_hyperparameters()

    def forward(self, images):
        n_bands = images.size(1) // 2
        reference, testimg = images[:, 0:n_bands], images[:, n_bands:2*n_bands]
        reference = reference.to(self.device).float()
        testimg = testimg.to(self.device).float()
        return self.change_detector(reference, testimg)

    def training_step(self, batch, batch_idx):
        y_hat = self(batch['image']).squeeze(1)
        loss = self.criterion(y_hat, batch['mask'].to(self.device).float())
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat = self(batch['image']).squeeze(1)
        loss = self.criterion(y_hat, batch['mask'].to(self.device).float())
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        y_hat = self(batch['image']).squeeze(1)
        loss = self.criterion(y_hat, batch['mask'].to(self.device).float())
        self.log('test_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.003,
                                      weight_decay=0.009449677083344786, amsgrad=False)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        return {"optimizer": optimizer, "lr_scheduler": None}