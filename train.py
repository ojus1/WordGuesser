import torch
import pytorch_lightning as pl
from core.config import cfg
from core.model import Net, loss_fn
from core.tokenizer import get_num_tokens_padding_idx, CharLevelTokenizer
from core.WordDataModule import WordDataModule
import os
from torchmetrics import F1, Accuracy
from argparse import ArgumentParser


class Experiment(pl.LightningModule):
    def __init__(self, model, padding_idx, args):
        super().__init__()

        self.cfg = cfg
        if not args.cfg is None:
            self.cfg.merge_from_file(args.cfg)
        self.cfg.freeze()
        print(self.cfg)

        self.padding_idx = padding_idx
        self.model = model

        self.train_f1 = F1(
            num_classes=num_tokens, ignore_index=padding_idx, mdmc_average="global"
        )
        self.val_f1 = F1(
            num_classes=num_tokens, ignore_index=padding_idx, mdmc_average="global"
        )

        self.train_acc = F1(
            num_classes=num_tokens, ignore_index=padding_idx, mdmc_average="global"
        )
        self.val_acc = F1(
            num_classes=num_tokens, ignore_index=padding_idx, mdmc_average="global"
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        seq, _, label = batch
        pred = self.model(seq)
        loss = loss_fn(pred, label, padding_idx=self.padding_idx)

        self.train_acc(pred.argmax(-1), label)
        self.log("ACC/train", self.train_acc, prog_bar=True)
        self.train_f1(pred.argmax(-1), label)
        self.log("F1/train", self.train_f1)
        self.log("LOSS/train", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        seq, _, label = batch
        pred = self.model(seq)
        loss = loss_fn(pred, label, padding_idx=self.padding_idx)

        self.val_acc(pred.argmax(-1), label)
        self.log("ACC/val", self.val_acc, prog_bar=True)
        self.val_f1(pred.argmax(-1), label)
        self.log("F1/val", self.val_f1)
        self.log("LOSS/val", loss)

        return loss

    def configure_optimizers(self):
        lr = self.cfg.MODEL.learning_rate
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)

        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=320)

        return [opt], [schedule]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # TODO: add arguments
        parser.add_argument(
            "--learning_rate", type=float, default=2e-3, help="adam: learning rate"
        )
        return parser


from pytorch_lightning.callbacks import ModelCheckpoint

if __name__ == "__main__":
    num_tokens, padding_idx = get_num_tokens_padding_idx("./data/tokenizer.pth")
    gru = Net(num_tokens, padding_idx)

    parser = ArgumentParser()
    parser.add_argument(
        "--dataset_csv_path",
        default="./data/train.csv",
        help="The csv must have two columns, words and their description. Default: uses open-source english dictionary.",
    )
    parser.add_argument(
        "--cfg",
        help="Path to YAML file containing hparams. Uses default values if not provided.",
    )

    script_args, _ = parser.parse_known_args()
    # os.system(f"python3 build_tokenizer.py --dataset_csv_path {script_args.dataset_csv_path}")

    parser = WordDataModule.add_argparse_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = Experiment.add_model_specific_args(parser)
    args, _ = parser.parse_known_args()

    tokenizer = torch.load("./data/tokenizer.pth")
    dm = WordDataModule(tokenizer, script_args.dataset_csv_path)

    experiment = Experiment(gru, padding_idx, args)
    checkpointer = ModelCheckpoint(monitor="LOSS/val", mode="min")
    trainer = pl.Trainer.from_argparse_args(
        args,
        max_epochs=300,
        precision=16,
        stochastic_weight_avg=True,
        gpus=1,
        val_check_interval=0.3,
        callbacks=[checkpointer],
    )
    trainer.fit(experiment, dm)
