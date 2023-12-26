import copy
import datetime
import logging
import math
import os
import pprint
import random
import sys
import time
import traceback
import warnings
from logging import handlers

import numpy as np
import pandas as pd
# deep learning package
import torch
from datasets import Dataset
from omegaconf import DictConfig, OmegaConf, ListConfig
# from pytorch_lightning.utilities import rank_zero_only
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_everything_v1(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def seed_everything_v2(seed=0):
    import random
    import numpy as np
    import os
    import torch
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def adjust_learning_rate(optimizer, epoch, args):
    """
    :param optimizer: SGD optimizer
    :param epoch: current epoch
    :param args: args
    :return:
    Decay the learning rate based on schedule
    """

    lr = args.learning_rate
    if args.cos == 1:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    elif args.cos == 2:
        lr *= math.cos(math.pi * epoch / (args.epochs * 2))
    else:  # stepwise lr schedule
        lr = args.learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def print_time_delta(desc, start_ts):
    if start_ts is not None:
        now = time.time()
        delta = now - start_ts[0]
        start_ts[0] = now
        print(f'{desc} cost: {delta}')
        print(desc, '%.3f' % delta)


def print_error_info(e):
    print("str(Exception):\t", str(Exception))
    print("str(e):\t\t", str(e))
    print("repr(e):\t", repr(e))
    # Get information about the exception that is currently being handled
    exc_type, exc_value, exc_traceback = sys.exc_info()
    print("e.message:\t", exc_value)
    print(
        "Note, object e and exc of Class %s is %s the same."
        % (type(exc_value), ("not", "")[exc_value is e])
    )
    print("traceback.print_exc(): ", traceback.print_exc())
    print("traceback.format_exc():\n%s" % traceback.format_exc())


def pp(input_obj):
    pprint.pprint(input_obj)


def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger = logging.getLogger(name)
    logger.setLevel(level)

    for level in (
            "debug",
            "info",
            "warning",
            "error",
            "exception",
            "fatal",
            "critical",
    ):
        setattr(logger, level, getattr(logger, level))

    return logger


log = get_logger(__name__)

class Result(dict):
    def __getattr__(self, name):
        return self[name]

    def __init__(self, *args, **kwargs):
        super(Result, self).__init__()
        for arg in args:
            for key, value in arg.items():
                self[key] = value
        self.add(**kwargs)

    # 序列化时调用
    def __getstate__(self):
        return None

    def add(self, **kwargs):
        for k, v in kwargs.items():
            self[k] = v

    def delete(self, keys):
        for k in keys:
            self.pop(k)

    def merge(self, merge_dict):
        if not isinstance(merge_dict, Result):
            raise TypeError("不支持的合并类型")
        for k, v in merge_dict.items():
            if k in ["msg", "status"] or k in self:
                continue
            self[k] = v

    def merge_or_update(self, merge_dict):
        if not isinstance(merge_dict, Result) and not isinstance(merge_dict, dict):
            raise TypeError("不支持的合并类型")
        for k, v in merge_dict.items():
            if k in ["msg", "status"]:
                continue
            self[k] = v

    @staticmethod
    def create_error_msg_result(msg="Error Result", **kwargs):
        result = Result()
        result["msg"] = msg
        result["status"] = False
        result.add(**kwargs)
        return result

    def get(self, name, other=None):
        if name is None:
            return list(self.values())
        elif isinstance(name, str):
            return self[name] if name in self else other
        elif isinstance(name, list):
            values = [self[n] for n in name]
            return values
        else:
            return self.create_error_msg_result(msg=f"Key值类型{type(name)}不支持")

    def print(self, name=None):
        pp("  =====" + self["msg"] + "=====")
        values = self.get(name)
        if name is None:
            name = list(self.keys())
        for i, k in enumerate(name):
            v = values[i]
            pp(f"  {k}:    {v}")
        pp("  =====" + self["msg"] + "=====")

    def flatten_to_print(self):
        value_str = ""
        keys = self.keys()
        for i, k in enumerate(keys):
            v = self[k]
            value_str = value_str + k + " : " + str(v) + "\n\n"
        return value_str

    def append_values(self, next_dict):
        if not isinstance(next_dict, Result) and not isinstance(next_dict, dict):
            raise TypeError("不支持的合并类型")
        for key in next_dict.keys():
            if key not in self.keys():
                self[key] = []

            self[key].append(next_dict[key]) if isinstance(self[key], list) else [
                self[key]
            ].append(next_dict[key])

    def str(self, key_name, default_value=""):
        return self.get(key_name, default_value)

    def bool(self, key_name, default_value=False):
        return self.get(key_name, default_value)

    def int(self, key_name, default_value=0):
        return self.get(key_name, default_value)

    def float(self, key_name, default_value=0.0):
        return self.get(key_name, default_value)

    def list(self, key_name, default_value=[]):
        return self.get(key_name, default_value)

    def dict(self, key_name, default_value={}):
        return self.get(key_name, default_value)

    def set(self, key_name, value):
        self[key_name] = value

    def set_with_dict(self, dict_value):
        for key, value in dict_value.items():
            if "." in key:
                key_list = key.split(".")
                self[key_list[0]][key_list[1]] = value
            else:
                self[key] = value

    def __deepcopy__(self, memo=None, _nil=[]):
        if memo is None:
            memo = {}
        d = id(self)
        y = memo.get(d, _nil)
        if y is not _nil:
            return y

        dict = Result()
        memo[d] = id(dict)
        for key in self.keys():
            dict.__setattr__(
                copy.deepcopy(key, memo), copy.deepcopy(
                    self.__getattr__(key), memo)
            )
        return dict

    def copy(self):
        return super().copy()


def check_config(config: DictConfig):
    OmegaConf.set_struct(config, False)

    # disable python warnings if <config.ignore_warnings=True>
    if config.ignore_warnings:
        warnings.filterwarnings("ignore")

    config.lr = float(config.lr)

    for k, v in config.items():
        if v == 'True' or v == 'true':
            v = True
            print(k, type(v), v)
        elif v == 'False' or v == 'false':
            v = False
            print(k, type(v), v)
        else:
            v = v
        config[k] = v
    # config.dataset_processor = config.dataset + '.' + config.dataset_processor

    # config.cache_dir = config.cache_dir + config.pretrain_model.split(":")[-1]
    task_save_name = config.comet_name

    task_save_name += f"__{config.dataset}__{config.arch}"

    config.task_full_name = f"{task_save_name}__batch{config.batch_size}__seed{config.seed}__{datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    config.task_full_name = config.task_full_name.replace("-", "_")

    print(f'fp16: {type(config.fp16)}, {config.fp16}')
    # 设置cuda
    if not config.use_gpu:
        # 不使用gpu
        config.default_device = "cpu"
        config.want_gpu_num = 0
        config.visible_cuda = None
    else:
        # 使用gpu
        # if config.wait_gpus:
        config.want_gpu_num = (
            int(config.visible_cuda.split("auto_select_")[-1])
            if "auto_select_" in str(config.visible_cuda)
            else len(config.visible_cuda)
        )
        config.default_device = f"cpu"
        if not config.wait_gpus:
            if "auto_select_" not in str(config.visible_cuda):
                gpus = config.visible_cuda.split(",") if not isinstance(
                    config.visible_cuda, ListConfig) else config.visible_cuda
                if isinstance(gpus, int):
                    config.want_gpu_num = 1
                    config.default_device = f"cuda:{gpus}"
                else:
                    config.want_gpu_num = len(gpus)
                    config.default_device = f"cuda:{gpus[0]}"

    return config


# 获取异常函数及行号
def print_error_info():
    """Return the frame object for the caller's stack frame."""
    try:
        raise Exception
    except:
        f = sys.exc_info()[2].tb_frame.f_back
    print(f.f_code.co_name, f.f_lineno)


def get_eval_metrics(outputs, config):
    """
    评价指标计算
    :param config:
    :param outputs: Dataframe类型,必须要包含的column为 [generated, reference, other_features, input_ids, labels]
    :return: dict
    """
    if not isinstance(outputs, Dataset):
        try:
            if isinstance(outputs, pd.DataFrame):
                outputs = Dataset.from_pandas(outputs)
            elif isinstance(outputs, dict):
                outputs = Dataset.from_dict(outputs)
            else:
                raise ValueError()
        except Exception as e:
            raise ValueError("评价指标计算的输入必须是Dataset, pd.DataFrame 或者 dict类型")

    result = Result()
    eval_metrics = config.eval_metrics

    ###############################################
    # 计算 Macro-F1
    ###############################################
    if "f1" in eval_metrics:
        log.info("计算 f1 score ing...")
        f1score = f1_score(outputs['labels'], outputs['preds'], average='macro')
        result.add(f1score=f1score)
        log.info(f"F1score = {str(f1score)}")

    ###############################################
    # 计算 ACC
    ###############################################
    if "accuracy" in eval_metrics:
        log.info("计算 accuracy ing...")
        acc = accuracy_score(outputs['labels'], outputs['preds'])
        result.add(acc=acc)
        log.info(f"acc = {str(acc)}")

    if "precision" in eval_metrics:
        log.info("计算 precision ing...")
        precision = precision_score(outputs['labels'], outputs['preds'], average='macro')
        result.add(precision=precision)
        log.info(f"precision = {str(precision)}")

    if "recall" in eval_metrics:
        log.info("计算 recall ing...")
        recall = recall_score(outputs['labels'], outputs['preds'], average='macro')
        result.add(recall=recall)
        log.info(f"result = {str(result)}")

    # if "auc" in eval_metrics:
    #     log.info("计算 auc ing...")
    #     auc = roc_auc_score(outputs['labels'], outputs['preds'], average='macro')
    #     result.add(auc=auc)
    #     log.info(f"result = {str(result)}")

    matrix_set = (f1score, acc, precision, recall)  # , auc
    return result, matrix_set


def get_scheduler(optimizer, min_lr):
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1, eta_min=min_lr, last_epoch=-1)
    return scheduler
