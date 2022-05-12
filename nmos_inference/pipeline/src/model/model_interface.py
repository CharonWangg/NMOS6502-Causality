import inspect
import torch
import numpy as np
import importlib
from torch.nn import functional as F
import torch.optim.lr_scheduler as lrs
import pytorch_lightning as pl
from ..utils.model_util import diy_loss, diy_optimizer
from sklearn.metrics import roc_auc_score
from torchmetrics.functional import accuracy, auroc


class ModelInterface(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        # self.__dict__.update(kwargs)
        self.save_hyperparameters()
        self.load_model()
        self.configure_loss()

    def forward(self, img):
        return self.model(img)

    def training_step(self, batch, batch_idx):
        # img, labels, filename = batch
        img, labels = batch
        out = self(img)
        # get the prediction label
        label_digit = labels[:, 0]
        out_digit = out.mean(dim=-2).softmax(dim=-1)
        loss = self.loss_function(out, labels)
        acc = accuracy(out_digit, label_digit)
        auc = auroc(out_digit, label_digit, num_classes=self.hparams.num_classes)
        # Will add _epoch when both on_step and on_epoch are True
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_auc', auc, on_step=True, on_epoch=True, prog_bar=True)
        loss += self.hparams.l1 * self.l1_norm() + self.hparams.l2 * self.l2_norm()
        return loss

    def validation_step(self, batch, batch_idx):
        # img, labels, filename = batch
        img, labels = batch
        out = self(img)
        label_digit = labels[:, 0]
        out_digit = out.mean(dim=-2).softmax(dim=-1).argmax(dim=-1)
        loss = self.loss_function(out, labels)
        # get the prediction label
        correct_num = sum(label_digit == out_digit).cpu().item()
        total = label_digit.size(0)

        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=False)

        return (np.mean(out.cpu().numpy(), axis=1),
                np.mean(labels.cpu().numpy(), axis=1),
                correct_num,
                total,
                loss.item(),
                )

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        img, labels = batch
        out = self(img)
        loss = self.loss_function(out, labels)
        label_digit = labels[:, 0]
        # get the prediction label
        out_digit = out.mean(dim=-2).softmax(dim=-1).argmax(dim=-1)
        correct_num = sum(label_digit == out_digit).cpu().item()
        total = label_digit.size(0)

        self.log('test_loss', loss, on_step=True, on_epoch=False, prog_bar=True)

        return (np.mean(out.cpu().numpy(), axis=1),
                np.mean(labels.cpu().numpy(), axis=1),
                correct_num,
                total,
                loss.item(),
                )

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        img, labels = batch
        out = self(img)
        # get the prediction label
        out_digit = out.mean(dim=-2).softmax(dim=-1)
        # print(out_digit.shape)
        return out_digit.cpu().numpy()

    # def training_epoch_end(self, train_step_outputs):
    #     # outputs is a list of output from training_step
    #     loss = torch.stack([x for x in train_step_outputs]).mean()
    #     print(f'train_loss: {loss}')

    def validation_epoch_end(self, validation_step_outputs):
        # outputs is a list of output from validation_step
        # pred: batch_idx * (batch_size, num_classes)
        # label: batch_idx * (1, batch_size)
        preds = np.vstack([x[0] for x in validation_step_outputs])
        labels = np.hstack([x[1] for x in validation_step_outputs])
        correct_num = sum([x[2] for x in validation_step_outputs])
        total = sum([x[3] for x in validation_step_outputs])
        loss = sum([x[4] for x in validation_step_outputs]) / len(validation_step_outputs)

        val_acc = correct_num / total
        val_auc = roc_auc_score(labels, preds[:, 1])

        self.log('val_epoch_loss', loss, on_epoch=True, prog_bar=False)
        self.log('val_epoch_acc', val_acc, on_epoch=True, prog_bar=False)
        self.log('val_epoch_auc', val_auc, on_epoch=True, prog_bar=False)

    def test_epoch_end(self, test_step_outputs):
        preds = np.vstack([x[0] for x in test_step_outputs])
        labels = np.hstack([x[1] for x in test_step_outputs])
        correct_num = sum([x[2] for x in test_step_outputs])
        total = sum([x[3] for x in test_step_outputs])
        loss = sum([x[4] for x in test_step_outputs]) / len(test_step_outputs)

        test_acc = correct_num / total
        test_auc = roc_auc_score(labels, preds[:, 1])

        self.log('test_epoch_loss', loss, on_epoch=True, prog_bar=False)
        self.log('test_epoch_acc', test_acc, on_epoch=True, prog_bar=False)
        self.log('test_epoch_auc', test_auc, on_epoch=True, prog_bar=False)



    def l1_norm(self):
        return sum(p.abs().sum() for p in self.model.parameters() if p.ndim >= 2)

    def l2_norm(self):
        return sum((p ** 2).sum() for p in self.model.parameters() if p.ndim >= 2)

    def nuc_norm(self):
        return sum(torch.norm(p, p="nuc") for p in self.model.parameters() if p.ndim >= 2)

    def configure_optimizers(self):
        # optimizer
        if self.hparams.optimizer.lower() == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(),
                                         lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer.lower() == 'adamw':
            optimizer = torch.optim.AdamW(self.model.parameters(),
                                          lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer.lower() == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(),
                                        lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer.lower() == 'rmsprop':
            optimizer = torch.optim.RMSprop(self.model.parameters(),
                                            lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer.lower() == 'diy':
            optimizer = diy_optimizer(self.model,
                                      lr=self.lr, weight_decay=self.hparams.weight_decay,
                                      epsilon=self.hparams.epsilon, momentum=self.hparams.momentum,
                                      correct_bias=self.hparams.correct_bias)
        else:
            raise ValueError("Unknown optimizer")

        # scheduler
        if self.hparams.lr_scheduler.lower() == 'cyclic':
            # TODO: add cyclic scheduler
            scheduler = {"scheduler": lrs.CyclicLR(optimizer,
                                                   base_lr=self.hparams.lr_decay_min_lr,
                                                   max_lr=self.hparams.lr,
                                                   step_size_up=self.hparams.lr_decay_steps,
                                                   step_size_down=self.hparams.lr_decay_steps,
                                                   mode=self.hparams.lr_decay_mode),
                         "interval": "step"}
        elif self.hparams.lr_scheduler.lower() == 'cosine':
            scheduler = {"scheduler": lrs.CosineAnnealingLR(optimizer,
                                                            T_max=self.hparams.max_epochs),
                         "interval": "step"}
        elif self.hparams.lr_scheduler.lower() == 'plateau':
            # TODO: add plateau scheduler
            scheduler = lrs.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        elif self.hparams.lr_scheduler.lower() == 'step':
            scheduler = {"scheduler": lrs.StepLR(optimizer,
                                                 step_size=self.hparams.lr_decay_steps,
                                                 gamma=self.hparams.lr_decay_rate),
                         "interval": "step"}
        elif self.hparams.lr_scheduler.lower() == 'multistep':
            scheduler = {"scheduler": lrs.MultiStepLR(optimizer,
                                                      milestones=[135, 185],
                                                      gamma=self.hparams.lr_decay_rate),
                         "interval": "epoch"}
        elif self.hparams.lr_scheduler.lower() == 'one_cycle':
            scheduler = {"scheduler": lrs.OneCycleLR(optimizer,
                                                     max_lr=self.hparams.lr,
                                                     steps_per_epoch=self.hparams.lr_decay_steps,
                                                     epochs=self.hparams.max_epochs,
                                                     anneal_strategy='linear',
                                                     div_factor=self.hparams.max_epochs,
                                                     final_div_factor=self.hparams.max_epochs,
                                                     verbose=True
                                                     ),
                         "interval": "step"}
        elif self.hparams.lr_scheduler.lower() == 'nmos':
            steps_per_epoch = (50000 // self.hparams.train_batch_size) + 1

            scheduler = {"scheduler": lrs.OneCycleLR(optimizer,
                                                     max_lr=self.hparams.lr,
                                                     epochs=self.hparams.max_epochs,
                                                     steps_per_epoch=steps_per_epoch,
                                                     ),
                         "interval": "step"}
        elif self.hparams.lr_scheduler.lower() == 'constant':
            scheduler = {"scheduler": lrs.ConstantLR(optimizer),
                         "interval": "step"}
        else:
            raise ValueError("Unknown scheduler")

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def configure_loss(self):
        loss = self.hparams.loss
        if loss == 'mse':
            self.loss_function = F.mse_loss
        elif loss == 'l1':
            self.loss_function = F.l1_loss
        elif loss == 'cross_entropy':
            self.loss_function = F.cross_entropy
        elif loss == 'binary_cross_entropy':
            self.loss_function = F.binary_cross_entropy
        elif loss == 'diy':
            # calculate loss in the model class
            self.loss_function = diy_loss
        else:
            raise ValueError("Invalid Loss Type!")

    def load_model(self):
        name = self.hparams.model_name
        # Change the `snake_case.py` file name to `CamelCase` class name.
        # Please always name your model file name as `snake_case.py` and
        # class name corresponding `CamelCase`.
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        try:
            Model = getattr(importlib.import_module(
                '.' + name, package=__package__), camel_name)
        except:
            raise ValueError(
                f'Invalid Module File Name or Invalid Class Name {name}.{camel_name}!')
        self.model = self.instancialize(Model)

    def instancialize(self, Model, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        class_args = inspect.getfullargspec(Model.__init__).args[1:]
        inkeys = self.hparams.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams, arg)
        args1.update(other_args)
        return Model(**args1)
