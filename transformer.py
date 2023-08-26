from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
import math
from torch.utils.data import DataLoader
from timeit import default_timer as timer
from myIterDataPipe import MapperIterDataPipe
import preprocessing as pre
from torchtext.data.metrics import bleu_score
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)

# 代码词汇表的大小
CODE_VOCAB_SIZE = len(pre.vocab_transform[pre.CODE_LANGUAGE])
print("代码词汇：",CODE_VOCAB_SIZE)
# 自然语言词汇表的大小
NL_VOCAB_SIZE = len(pre.vocab_transform[pre.NL_LANGUAGE])
print("nl词汇：",NL_VOCAB_SIZE)
# 嵌入向量的维度大小
EMB_SIZE = 240
# 多头注意力机制中的头数
NHEAD = 6
# 前馈神经网络中的隐藏层维度
FFN_HID_DIM = 100
# 批大小
BATCH_SIZE = 10
# 编码器层数
NUM_ENCODER_LAYERS = 3
# 解码器层数
NUM_DECODER_LAYERS = 3
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3


# 将位置编码添加到嵌入中
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        tem = self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])
        return tem

# 将输入索引转换为相应的标记嵌入向量
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        # 对嵌入向量进行缩放
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)
    
# Transformer类
class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        # 线性层，将Transformer的输出映射到目标语言的词汇表大小
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        # Embedding层 索引序列转换为嵌入向量
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        # 位置编码层，为嵌入向量添加位置信息
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        # 做源序列嵌入
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        # 目标序列嵌入
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        # transformer层
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    # 编码
    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask)

    # 解码
    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory,
                          tgt_mask)

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len).type(torch.bool)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=DEVICE).type(torch.bool)
    
    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, CODE_VOCAB_SIZE, NL_VOCAB_SIZE, FFN_HID_DIM)

# 使用Xavier均匀分布初始化方法对模型的参数进行初始化
for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

#指定GPU
transformer = transformer.to(DEVICE)
# 损失函数
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
#adam优化器
optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

def train_epoch(model, optimizer):
    model.train()
    losses = 0
    train_iter = MapperIterDataPipe(split='train')
    train_dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE, collate_fn=pre.collate_fn)
    mindex = 0
    for src, tgt in train_dataloader:
        mindex+=1
        if mindex%10000 == 0: print(mindex)
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)
        tgt_input = tgt[:-1, :]
        # mask
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
        # Forward
        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)
        # Buffer reset
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        tgt_out = tgt[1:, :]
        # Calculating the loss
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        # Back
        loss.backward()
        # Updating model parameters
        optimizer.step()
        # Calculate the total loss
        losses += loss.item()
    return losses / len(list(train_dataloader))

def evaluate(model):
    model.eval()
    losses = 0
    val_iter = MapperIterDataPipe(split='valid', language_pair=(pre.CODE_LANGUAGE, pre.NL_LANGUAGE))
    val_dataloader = DataLoader(val_iter, batch_size=BATCH_SIZE, collate_fn=pre.collate_fn)
    for src, tgt in val_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)
        tgt_input = tgt[:-1, :]
        # 掩码
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
        # 前向传播
        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)
        torch.cuda.empty_cache()
        tgt_out = tgt[1:, :]
        # 计算损失
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        # 计算总损失
        losses += loss.item()


    return losses / len(list(val_dataloader))

# 用贪婪算法生成输出序列
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)
    # 编码
    memory = model.encode(src, src_mask)
    # 初始化输出
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len-1):
        memory = memory.to(DEVICE)
        # 目标掩码
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(DEVICE)
        # 解码
        out = model.decode(ys, memory, tgt_mask)
        # 转置
        out = out.transpose(0, 1)
        # 得到每个单词在目标词汇表上的概率分布
        prob = model.generator(out[:, -1])
        # 寻找概率最高符号
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()
        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys
transformer.load_state_dict(torch.load('model.pth'))
print("总参")
print(sum(p.numel() for p in transformer.parameters() if p.requires_grad))
NUM_EPOCHS = 1
'''
for epoch in range(1, NUM_EPOCHS+1):
    start_time = timer()
    train_loss = train_epoch(transformer, optimizer)
    end_time = timer()
    #val_loss = evaluate(transformer)
    print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: , "f"Epoch time = {(end_time - start_time):.3f}s"))
torch.save(transformer.state_dict(), 'model.pth')
'''
def translate(model: torch.nn.Module, src_sentence: str):
    model.eval()
    src = pre.text_transform[pre.CODE_LANGUAGE](src_sentence).view(-1, 1)
    # token数
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model,  src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX).flatten()
    return " ".join(pre.vocab_transform[pre.NL_LANGUAGE].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")

print(translate(transformer,  ["def", "save","policy", "(", "self", ",", "path", ")", ":", "with", "open", "(", "path", ",", "'wb'", ")", "as", "f", ":", "pickle", ".", "dump", "(", "self", ".", "policy", ",", "f", ")"]))
'''
miter = MapperIterDataPipe(split='valid', language_pair=(pre.CODE_LANGUAGE, pre.NL_LANGUAGE))
tem = pre.yield_tokens(miter,pre.CODE_LANGUAGE)
for item in tem:
    print(translate(transformer,item))
    break'''