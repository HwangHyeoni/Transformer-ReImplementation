
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

class CustomDataLoader:

    def __init__(self, de_tokenizer, en_tokenizer, bos_idx, eos_idx, pad_idx, batch_size):
        self.de_tokenizer = de_tokenizer
        self.en_tokenizer = en_tokenizer
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.batch_size = batch_size

    def generate_batch(self, data_batch):
        de_batch, en_batch = [], []
        for (de_item, en_item) in data_batch:
            de_batch.append(torch.cat([torch.tensor([self.bos_idx]), de_item, torch.tensor([self.eos_idx])], dim=0))
            en_batch.append(torch.cat([torch.tensor([self.bos_idx]), en_item, torch.tensor([self.eos_idx])], dim=0))
        de_batch = pad_sequence(de_batch, padding_value=self.pad_idx)
        en_batch = pad_sequence(en_batch, padding_value=self.pad_idx)
        return de_batch, en_batch

    def make_iter(self, train_data, val_data, test_data):


        train_iter = DataLoader(train_data, batch_size=self.batch_size,
                                shuffle=True, collate_fn=self.generate_batch)
        valid_iter = DataLoader(val_data, batch_size=self.batch_size,
                                shuffle=True, collate_fn=self.generate_batch)
        test_iter = DataLoader(test_data, batch_size=self.batch_size,
                            shuffle=True, collate_fn=self.generate_batch)
    
        return train_iter, valid_iter, test_iter