from data import de_vocab,en_vocab

INPUT_DIM = len(de_vocab)
OUTPUT_DIM = len(de_vocab)
HIDDEN_DIM =256
ENC_LAYERS = 6
DEC_LAYERS = 6
ENC_HEADS = 8
DEC_HEADS = 8
ENC_PF_DIM = 512
DEC_PF_DIM = 512
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1
SRC_PAD_IDX = de_vocab['<pad>']
TRG_PAD_IDX = en_vocab['<pad>']