import lightning as pl
from sklearn.metrics import precision_recall_fscore_support as prfs
from .utils.helpers import (get_loaders, get_criterion,
                           load_model, initialize_metrics, get_mean_metrics,
                           set_metrics)
import torch

class LSNetLightning(pl.LightningModule):
    def __init__(self, opt):
        super(LSNetLightning, self).__init__()
        self.opt = opt
        self.model = load_model(opt, self.device)
        self.criterion = get_criterion(opt)
        self.best_metrics = {'cd_f1scores': -1, 'cd_recalls': -1, 'cd_precisions': -1}

    def forward(self, batch):
        img1, img2 = self._split_images(batch['image'])
        img_cat = torch.cat([img1, img2],dim=0)
        return self.model(img_cat)

    def _split_images(self, images, bands='rgb'):
        n_bands = 3 if bands == "rgb" else 13
        pre, post = images[:, 0:n_bands], images[:, n_bands:2*n_bands]
        return pre, post

    def _log_metrics(self, metrics_dict, cd_preds, mask, cd_loss):
        cd_preds = cd_preds[-1]
        _, cd_preds = torch.max(cd_preds, 1)
        cd_corrects = (100 *
                       (cd_preds.squeeze().byte() == mask.squeeze().byte()).sum() /
                       (mask.size()[0] * (self.opt.patch_size ** 2)))

        cd_train_report = prfs(mask.data.cpu().numpy().flatten(),
                               cd_preds.data.cpu().numpy().flatten(),
                               average='binary',
                               pos_label=1)

        metrics_dict = set_metrics(metrics_dict,
                            cd_loss,
                            cd_corrects,
                            cd_train_report,
                            self.scheduler.get_last_lr())

    def on_train_epoch_start(self):
        self.train_metrics = initialize_metrics()

    def training_step(self, batch, batch_idx):
        cd_preds = self(batch)
        mask = batch['mask']
        bce_loss, dice_loss, cd_loss = self.criterion([cd_preds[-1]], mask)
        loss = cd_loss
        self._log_metrics(self.train_metrics, cd_preds, mask, loss)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        mean_train_metrics = get_mean_metrics(self.train_metrics)
        self.log_dict({f'train/{k}': v for k,v in mean_train_metrics.items()})

    def on_validation_epoch_start(self):
        self.val_metrics = initialize_metrics()

    def validation_step(self, batch, batch_idx):
        cd_preds = self(batch)
        mask = batch['mask']
        bce_loss, dice_loss, cd_loss = self.criterion([cd_preds[-1]], mask)
        loss = cd_loss
        self._log_metrics(self.val_metrics, cd_preds, mask, loss)
        self.log('val_loss', loss, on_epoch=True)
        return loss

    def on_validation_epoch_end(self):
        mean_train_metrics = get_mean_metrics(self.val_metrics)
        self.log_dict({f'val/{k}': v for k,v in mean_train_metrics.items()})

    def on_test_epoch_start(self):
        self.test_metrics = initialize_metrics()

    def test_step(self, batch, batch_idx):
        cd_preds = self(batch)
        mask = batch['mask']
        bce_loss, dice_loss, cd_loss = self.criterion([cd_preds[-1]], mask)
        loss = cd_loss
        self._log_metrics(self.test_metrics, cd_preds, mask, loss)
        self.log('test_loss', loss, on_epoch=True)
        return loss

    def on_test_epoch_end(self):
        mean_train_metrics = get_mean_metrics(self.test_metrics)
        self.log_dict({f'test/{k}': v for k,v in mean_train_metrics.items()})

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.opt.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)
        return [optimizer], [self.scheduler]