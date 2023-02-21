from models import Encoder, Decoder, Transformer
from config import *
import logging
import torch
import torch.nn as nn
import argparse
import random
import numpy as np
import torch.optim as optim
import math
import time
from data import *
import spacy
import io
import wandb

logging.basicConfig(format='%(asctime)s -  %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#wandb init, config setting
wandb.init(project='transformer_1')
wandb.config = {
  "learning_rate": 1e-4,
  "epochs": 50,
  "batch_size": 64
}

def get_args():
    parser = argparse.ArgumentParser()

    # Run settings
    parser.add_argument('--max_seq_len',
                        default=50, type=int)
    parser.add_argument('--learning_rate',
                        default=1e-4, type=float)
    parser.add_argument('--epoch',
                        default=100, type=int)
    parser.add_argument('--batch',
                        default=64, type=int)
    parser.add_argument('--seed',
                        default=0, type=int)
    parser.add_argument('--mode',
                        default='train')

    args = parser.parse_args()
    logger.info(f"device: {device}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    return args

# 인코더(encoder)와 디코더(decoder) 객체 선언
enc = Encoder(INPUT_DIM, HIDDEN_DIM, ENC_LAYERS, ENC_HEADS, ENC_PF_DIM, ENC_DROPOUT, device)
dec = Decoder(OUTPUT_DIM, HIDDEN_DIM, DEC_LAYERS, DEC_HEADS, DEC_PF_DIM, DEC_DROPOUT, device)




def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

# 모델 학습 함수
def train(model, iterator, optimizer, criterion, clip):
        
    model.train()#학습 모드
    epoch_loss =0

    #전체 학습 데이터를 확인하며
    for i, (src, trg) in enumerate(iterator):
        src = src.T.to(device)
        trg = trg.T.to(device)


        optimizer.zero_grad()

        # 출력 단어의 마지막 인덱스(<eos>)제외
        # 입력할 떄는 <sos>부터 시작하도록 처리
        output, _ = model(src, trg[:,:-1])


        #output : [배치크기, trg_len-1, output_dim]
        #trg : [배치 크기, trg_len]

        output_dim = output.shape[-1]

        output = output.contiguous().view(-1, output_dim)
        # 출력단어의 인덱스 0 (<sos>)는 제외
        trg = trg[:,1:].contiguous().view(-1)

        #output : [배치 크기 * trg_len-1, output_dim]
        #trg : [배치 크기 * trg_len-1]

        #모델의 출력 결과와 타겟 문장을 비교해서 손실 계산
        loss = criterion(output, trg)
        loss.backward() #기울기 계산

        # 기울기 clipping 진행
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        #파라미터 업데이트
        optimizer.step()

        # 전체 손실값 계산
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)

# 모델 평가(evaluate) 함수
def evaluate(model, iterator, criterion):
    model.eval() # 평가 모드
    epoch_loss = 0

    with torch.no_grad():
        # 전체 평가 데이터를 확인하며
        for i, (src, trg) in enumerate(iterator):
            src = src.T.to(device)
            trg = trg.T.to(device)

            # 출력 단어의 마지막 인덱스(<eos>)는 제외
            # 입력을 할 때는 <sos>부터 시작하도록 처리
            output, _ = model(src, trg[:,:-1])


            # output: [배치 크기, trg_len - 1, output_dim]
            # trg: [배치 크기, trg_len]

            output_dim = output.shape[-1]

            output = output.contiguous().view(-1, output_dim)
            # 출력 단어의 인덱스 0(<sos>)은 제외
            trg = trg[:,1:].contiguous().view(-1)

            # output: [배치 크기 * trg_len - 1, output_dim]
            # trg: [배치 크기 * trg len - 1]

            # 모델의 출력 결과와 타겟 문장을 비교하여 손실 계산
            loss = criterion(output, trg)

            # 전체 손실 값 계산
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)

def main(args):
    if args.mode == "train":
        # Transformer 객체 선언
        model = Transformer(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)
        print(f'The model has {count_parameters(model):,} trainable parameters')

        #가중치 초기화
        model.apply(initialize_weights)

        #Adam optimizer
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

        criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)

        best_valid_loss = float('inf')
        for epoch in range(args.epoch):
            start_time = time.time() # 시작 시간 기록

            train_loss = train(model, train_iter, optimizer, criterion, clip=1)
            valid_loss = evaluate(model, valid_iter, criterion)
            
            end_time = time.time() # 종료 시간 기록
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            wandb.log({"train_loss":train_loss, "valid_loss":valid_loss})

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), './checkpoint/transformer_german_to_english.pt')
            print(f'--------------START---------------')
            print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):.3f}')
            print(f'\tValidation Loss: {valid_loss:.3f} | Validation PPL: {math.exp(valid_loss):.3f}')
        
        print(f'--------------END--------------')
    

    if args.mode == "test":
        model.load_state_dict(torch.load("/checkpoint/transformer_german_to_english.pt"))
        test_loss = evaluate(model, test_iter, criterion)

        print(f'Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):.3f}')

        # 번역(translation) 함수
        def translate_sentence(sentence, de_vocab, en_vocab, model, device, max_len=50, logging=True):
            model.eval() # 평가 모드

            if isinstance(sentence, str):
                nlp = spacy.load('de_core_news_sm')
                tokens = [token.text.lower() for token in nlp(sentence)]
            else:
                tokens = [token.lower() for token in sentence]

            # 처음에 <sos> 토큰, 마지막에 <eos> 토큰 붙이기
            tokens = ['<bos>'] + tokens + ['<eos>']
            if logging:
                print(f"전체 소스 토큰: {tokens}")

            src_indexes = [de_vocab[token] if token in de_vocab else de_vocab['<unk>'] for token in tokens]
            if logging:
                print(f"소스 문장 인덱스: {src_indexes}")

            src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)

            # 소스 문장에 따른 마스크 생성
            src_mask = model.make_src_mask(src_tensor)

            # 인코더(endocer)에 소스 문장을 넣어 출력 값 구하기
            with torch.no_grad():
                enc_src = model.encoder(src_tensor, src_mask)

            # 처음에는 <sos> 토큰 하나만 가지고 있도록 하기
            trg_indexes = [en_vocab['<bos>']]

            for i in range(max_len):
                trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

                # 출력 문장에 따른 마스크 생성
                trg_mask = model.make_trg_mask(trg_tensor)

                with torch.no_grad():
                    output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)

                # 출력 문장에서 가장 마지막 단어만 사용
                pred_token = output.argmax(2)[:,-1].item()
                trg_indexes.append(pred_token) # 출력 문장에 더하기

                # <eos>를 만나는 순간 끝
                if pred_token == en_vocab['<eos>']:
                    break

            # 각 출력 단어 인덱스를 실제 단어로 변환
            trg_tokens = [en_vocab.get_itos()[i] for i in trg_indexes]

            # <sos>와 <eos>는 제외하고 출력 문장 반환
            return trg_tokens[1:-2], attention

        test_src = iter(io.open(test_filepaths[0], encoding="utf8"))
        test_trg = iter(io.open(test_filepaths[1], encoding="utf8"))
        for i in range(10):
            src = next(test_src).rstrip()
            trg = next(test_trg).rstrip()

            print(f'소스 문장: {src}')
            print(f'타겟 문장: {trg}')

            translation, attention = translate_sentence(src, de_vocab, en_vocab, model, device, logging=True)

            print("모델 출력 결과:", " ".join(translation))

            print("--------------------------------")



if __name__ == "__main__":
    main(get_args())