# coding: UTF-8
import torch
from tqdm import tqdm
import time
from datetime import timedelta
import numpy as np
PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号

def build_dataset(config):
    def load_dataset(path, pad_size=32):
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                parts = lin.split(',')
                if len(parts) < 3:
                    continue
                event = parts[0]
                label = parts[1]
                content = parts[2]
                token = config.tokenizer.tokenize(content)
                token = ['[CLS]'] + token
                seq_len = len(token)
                mask = []
                token_ids = config.tokenizer.convert_tokens_to_ids(token)

                if pad_size:
                    if len(token) < pad_size:
                        mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                        token_ids += ([0] * (pad_size - len(token)))
                    else:
                        mask = [1] * pad_size
                        token_ids = token_ids[:pad_size]
                        seq_len = pad_size
                contents.append((token_ids, int(label), int(event), seq_len, mask ))
        return contents

    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)
    return train, dev, test

class DatasetIterater(object):
    def __init__(self, batches, batch_size, device,pad_size):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = len(batches) % batch_size != 0  # 通过直接计算是否有余数来确定self.residue的值
        self.index = 0
        self.device = device
        self.pad_size = pad_size  # 存储 pad_size 以备后用

    def _to_tensor(self, datas):
        # 将输入数据转换为 LongTensor 并移动到指定设备
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)  # 输入的句子
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)  # 标签
        event = torch.LongTensor([_[2] for _ in datas]).to(self.device)  # 事件

        # pad前的长度
        seq_len = torch.LongTensor([_[3] for _ in datas]).to(self.device)  # 序列长度
        mask = torch.LongTensor([_[4] for _ in datas]).to(self.device)  # mask

        # 确保 x 和 mask 是二维的 (batch_size, sequence_length)

        mask = mask.view(-1, self.pad_size)  # 确保 mask 是二维的

        return (x, seq_len, mask, event), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device, config.pad_size)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))
