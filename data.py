from utils.dataloader import CustomDataLoader
from utils.tokenizer import Tokenizer
from utils.data_processing import *
from torchtext.utils import download_from_url, extract_archive



url_base = 'https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/'
train_urls = ('train.de.gz', 'train.en.gz')
val_urls = ('val.de.gz', 'val.en.gz')
test_urls = ('test_2016_flickr.de.gz', 'test_2016_flickr.en.gz')

train_filepaths = [extract_archive(download_from_url(url_base + url))[0] for url in train_urls]
val_filepaths = [extract_archive(download_from_url(url_base + url))[0] for url in val_urls]
test_filepaths = [extract_archive(download_from_url(url_base + url))[0] for url in test_urls]

tokenizer = Tokenizer()

de_vocab = build_vocab(train_filepaths[0], tokenizer.de_tokenizer)
en_vocab = build_vocab(train_filepaths[1], tokenizer.en_tokenizer)


train_data = data_process(train_filepaths, de_vocab, en_vocab, tokenizer.de_tokenizer, tokenizer.en_tokenizer)
val_data = data_process(val_filepaths, de_vocab, en_vocab, tokenizer.de_tokenizer, tokenizer.en_tokenizer)
test_data = data_process(test_filepaths, de_vocab, en_vocab, tokenizer.de_tokenizer, tokenizer.en_tokenizer)

BATCH_SIZE = 128
PAD_IDX = de_vocab['<pad>']
BOS_IDX = de_vocab['<bos>']
EOS_IDX = de_vocab['<eos>']

loader = CustomDataLoader(de_tokenizer=tokenizer.de_tokenizer,
                          en_tokenizer=tokenizer.en_tokenizer,
                          bos_idx=BOS_IDX,
                          eos_idx=EOS_IDX,
                          pad_idx=PAD_IDX,
                          batch_size=BATCH_SIZE
                          )



train_iter, valid_iter, test_iter = loader.make_iter(train_data, val_data, test_data)