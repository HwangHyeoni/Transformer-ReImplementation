import torch
import torch.nn as nn

class MultiHeadAttentionLayer(nn.Module):
  def __init__(self, hidden_dim, n_heads, dropout_ratio, device):
    super().__init__()

    assert hidden_dim % n_heads == 0

    self.hidden_dim = hidden_dim #임베딩 차원
    self.n_heads = n_heads # 헤드의 개수, 서로 다른 어텐션 컨셉의 수
    self.head_dim = hidden_dim // n_heads # 각 헤드에서의 임베딩 차원

    self.fc_q = nn.Linear(hidden_dim, hidden_dim) # Query값에 적용될 FC레이어
    self.fc_k = nn.Linear(hidden_dim, hidden_dim) # Key값에 적용될 FC레이어
    self.fc_v = nn.Linear(hidden_dim, hidden_dim) # Value값에 적용될 FC레이어

    self.fc_o = nn.Linear(hidden_dim, hidden_dim)

    self.dropout = nn.Dropout(dropout_ratio)

    self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device) #scaled 에 해당

  def forward(self, query, key, value, mask = None):

    batch_size = query.shape[0]
    #batch_size = 32

    # query : [batch_size, query_len, hidden_dim] 배치 사이즈, 쿼리 길이(단어 개수), 단어 임베딩 크기
    # key : [batch_size, key_len, hidden_dim]
    # value : [batch_size, value_len, hidden_dim]

    Q = self.fc_q(query)
    K = self.fc_k(key)
    V = self.fc_v(value)

    # Q : [batch_size, query_len, hidden_dim] 배치 사이즈, 쿼리 길이(단어 개수), 단어 임베딩 크기
    # K : [batch_size, key_len, hidden_dim]
    # V : [batch_size, value_len, hidden_dim]

    # hidden_dim → n_heads X head_dim 형태로 변형
    # n_heads(h)개의 서로 다른 어텐션(attention) 컨셉을 학습하도록 유도
    Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
    K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
    V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

    #각 헤드마다 쿼리와 키를 곱하고 스케일로 나눠준다 Attention energy 계산 (softmax안 식)
    energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
    # energy : [batch_size, n_heads, query_len, key_len]

    #마스크를 사용하는 경우
    if mask is not None:
      # 마스크 값이 0인 부분을 -1e10으로 채우기
      energy = energy.masked_fill(mask==0, -1e10) #fillna와 비슷

    # 어텐션 스코어 계산 : 각 단어에 대한 확률값
    attention = torch.softmax(energy, dim=-1) #마지막 차원 기준

    # attention: [batch_size, n_heads, query_len, key_len]

    # 여기에서 Scaled Dot Product Attention을 계산, V값과 곱함, attention value matrix 생성
    x = torch.matmul(self.dropout(attention), V)

    # x: [batch_size, n_heads, query_len, head_dim]
    
    x = x.permute(0,2,1,3).contiguous() #텐서의 shape을 조작하는 과정에서 메모리 저장 상태가 변경되는 경우 다시 돌리기 위함

    # x: [batch_size, query_len, n_heads, head_dim]
   
    x = x.view(batch_size, -1, self.hidden_dim) # concat

    # x: [batch_size, query_len, head_dim]
    
    x = self.fc_o(x) #output linear함수를 거친다.

    return x, attention



class PositionwiseFeedforwardLayer(nn.Module):
  def __init__(self, hidden_dim, pf_dim, dropout_ratio):
    super().__init__()

    self.fc_1 = nn.Linear(hidden_dim, pf_dim)
    self.fc_2 = nn.Linear(pf_dim, hidden_dim)

    self.dropout = nn.Dropout(dropout_ratio)

  def forward(self,x):
    
    # x: [batch_size, seq_len, hidden_dim]

    x = self.dropout(torch.relu(self.fc_1(x)))


    # x: [batch_size, seq_len, pf_dim]

    x = self.fc_2(x)

    # x: [batch_size, seq_len, hidden_dim]

    return x


class EncoderLayer(nn.Module):
  def __init__(self, hidden_dim, n_heads, pf_dim, dropout_ratio, device):
    super().__init__()

    self.self_attn_layer_norm = nn.LayerNorm(hidden_dim)
    self.ff_layer_norm = nn.LayerNorm(hidden_dim)
    self.self_attention = MultiHeadAttentionLayer(hidden_dim, n_heads, dropout_ratio, device)
    self.positionwise_feedforward = PositionwiseFeedforwardLayer(hidden_dim, pf_dim, dropout_ratio)
    self.dropout = nn.Dropout(dropout_ratio)


    # 하나의 임베딩이 복제되어 Q, K, V로 입력되는 방식
  def forward(self, src, src_mask):

    # src : [batch_size, src_len, hidden_dim]
    # src_mask : [batch_size, src_len]


    # self attention
    # 필요한 경우 마스크 행렬을 이용해서 어텐션할 단어를 조정

    _src, _ = self.self_attention(src, src, src, src_mask) # forward kwargs

    # dropout, residual connection and layer norm
    src = self.self_attn_layer_norm(src + self.dropout(_src))


    # src: [batch_size, src_len, hidden_dim]

    # position-wise feedforward

    _src = self.positionwise_feedforward(src)

    # dropout, residual and layernorm

    src = self.ff_layer_norm(src + self.dropout(_src))

    # src: [batch_size, src_len, hidden_dim]
 
    return src

class DecoderLayer(nn.Module):
    def __init__(self, hidden_dim, n_heads, pf_dim, dropout_ratio, device):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hidden_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hidden_dim) 
        self.ff_layer_norm = nn.LayerNorm(hidden_dim)
        self.self_attention = MultiHeadAttentionLayer(hidden_dim, n_heads, dropout_ratio, device)  # 셀프 어텐션
        self.encoder_attention = MultiHeadAttentionLayer(hidden_dim, n_heads, dropout_ratio, device) # 인코더 디코더 어텐션
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hidden_dim, pf_dim, dropout_ratio)
        self.dropout = nn.Dropout(dropout_ratio)

    # 인코더의 출력 값(enc_src)을 어텐션(attention)하는 구조
    def forward(self, trg, enc_src, trg_mask, src_mask):

        # trg: [batch_size, trg_len, hidden_dim]
        # enc_src: [batch_size, src_len, hidden_dim]
        # trg_mask: [batch_size, trg_len]
        # src_mask: [batch_size, src_len]

        # self attention
        # 자기 자신에 대하여 어텐션(attention)
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)

        # dropout, residual connection and layer norm
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))

        # trg: [batch_size, trg_len, hidden_dim]

        # encoder attention
        # 디코더의 쿼리(Query)를 이용해 인코더를 어텐션(attention)
        # 쿼리는 디코더의 출력 단어 trg, 키와 밸류로 src를 이용
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)

        # dropout, residual connection and layer norm
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))

        # trg: [batch_size, trg_len, hidden_dim]

        # positionwise feedforward
        _trg = self.positionwise_feedforward(trg)

        # dropout, residual and layer norm
        trg = self.ff_layer_norm(trg + self.dropout(_trg))

        # trg: [batch_size, trg_len, hidden_dim]
        # attention: [batch_size, n_heads, trg_len, src_len]

        return trg, attention
     