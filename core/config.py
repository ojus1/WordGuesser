from yacs.config import CfgNode as CN

_C = CN()

_C.MODEL = CN()
_C.MODEL.num_layers = 6
_C.MODEL.drop_out = 0.1
_C.MODEL.embed_dim = 192
_C.MODEL.learning_rate = 0.0001

_C.TRAIN = CN()
_C.TRAIN.min_mask_ratio = 0.3
_C.TRAIN.max_mask_ratio = 0.5
_C.TRAIN.replace_prob = 0.9
_C.TRAIN.train_size = 0.9
_C.TRAIN.repeat_val_dataset = 10
_C.TRAIN.batch_size = 1024
_C.TRAIN.csv_path = "./data/train.csv"
_C.TRAIN.num_workers = 12


_C.TOKENIZATION = CN()
_C.TOKENIZATION.mask_token = "[MASK]"
_C.TOKENIZATION.sep_token = "[SEP]"
_C.TOKENIZATION.cls_token = "[CLS]"
_C.TOKENIZATION.pad_token = "[PAD]"
_C.TOKENIZATION.unknown_token = "[UNK]"
_C.TOKENIZATION.max_seq_length = 128

cfg = _C
