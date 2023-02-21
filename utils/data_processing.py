import torch
from collections import Counter
from torchtext.vocab import vocab
import io
import random



def build_vocab(filepath, tokenizer):
  counter = Counter()
  with io.open(filepath, encoding="utf8") as f:
    for string_ in f:
      counter.update(tokenizer(string_))
  return vocab(counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'])

def data_process(filepaths, src_vocab, trg_vocab, src_tokenizer, trg_tokenizer):
  raw_src_iter = iter(io.open(filepaths[0], encoding="utf8"))
  raw_trg_iter = iter(io.open(filepaths[1], encoding="utf8"))
  data = []
  for (raw_src, raw_trg) in zip(raw_src_iter, raw_trg_iter):
    src_tensor_ = torch.tensor([src_vocab[token] if token in src_vocab else src_vocab['<unk>'] for token in src_tokenizer(raw_src.rstrip())],
                            dtype=torch.long)
    trg_tensor_ = torch.tensor([trg_vocab[token] if token in trg_vocab else trg_vocab['<unk>'] for token in trg_tokenizer(raw_trg.rstrip())],
                            dtype=torch.long)
    data.append((src_tensor_, trg_tensor_))
  return data


def data_split(data, p):
    random.shuffle(data)

    data_1 = data[:int(len(data)*p)]
    data_2 = data[int(len(data)*p):]

    return data_1, data_2


# class TranslationDataset(Dataset):
#     def __init__(self, filepaths, src_tokenizer, trg_tokenizer):

#         self.filepaths = filepaths
#         self.src_vocab = build_vocab(filepaths[0], src_tokenizer)
#         self.trg_vocab = build_vocab(filepaths[0], src_tokenizer)
#         self.data = data_process(filepaths, self.src_vocab, self.trg_vocab, src_tokenizer, trg_tokenizer)

    

#     def __len__(self):
#         return len(self.data)
    

#     def __getitem__(self, idx):

#         self.text = self.data[idx]

#         return torch.tensor(self.text)