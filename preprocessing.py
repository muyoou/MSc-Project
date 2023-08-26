from torchtext.vocab import build_vocab_from_iterator
from typing import Iterable, List
import re
import torch
import sbt
from torch.nn.utils.rnn import pad_sequence
from myIterDataPipe import MapperIterDataPipe

# 批大小
BATCH_SIZE = 32

CODE_LANGUAGE = 'code'
NL_LANGUAGE = 'nl'
SBT_LANGUAGE = 'sbt'
IDS_LANGUAGE = 'ids'

# Place-holders
token_transform = {}
vocab_transform = {}

# 生成分词器函数
#token_transform[CODE_LANGUAGE] = get_tokenizer('spacy', language='de_core_news_sm')
#token_transform[NL_LANGUAGE] = get_tokenizer('spacy', language='en_core_web_sm')

# Split Camel Case Naming
def split_camel_case(list):
    for i in range(len(list)):
        pattern = r'^[A-z]+(?:[A-Z][a-z]*)*$'
        match = re.match(pattern, list[i])
        if match is not None:
            list[i] = re.findall(r'[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))|[a-z]+', list[i])
        else: list[i] = [list[i]]

# 替换字符串和数字
def replaceSaI (list):
    newList = []
    for item in list:
        if item.isdigit():
            newList.append("<int>")
        elif item.startswith('"') and item.endswith('"'): newList.append("<str>")
        elif item.startswith('r"') and item.endswith('"'): newList.append("<str>")
        elif item.startswith('f"') and item.endswith('"'): newList.append("<str>")
        elif item.startswith('u"') and item.endswith('"'): newList.append("<str>")
        elif item.startswith('b"') and item.endswith('"'): newList.append("<str>")
        elif item.startswith("'") and item.endswith("'"): newList.append("<str>")
        elif item.startswith("r'") and item.endswith("'"): newList.append("<str>")
        elif item.startswith("f'") and item.endswith("'"): newList.append("<str>")
        elif item.startswith("u'") and item.endswith("'"): newList.append("<str>")
        elif item.startswith("b'") and item.endswith("'"): newList.append("<str>")
        else:
            newList.append(item)
    return newList

def replaceEnd(item):
    newList = []
    while True:
        index = 0
        for ends in ['.',',',')',']',':']:
            if item.endswith(ends): 
                newList=[ends]+newList
                index+=1
                item = item[:-1]
        if index  == 0: break
    newList = [item]+newList
    return newList



# 将句子分为token
def yield_tokens(data_iter: Iterable, language: str) -> List[str]:
    language_index = {CODE_LANGUAGE: 0, NL_LANGUAGE: 1, SBT_LANGUAGE: 2,IDS_LANGUAGE: 2, 'spe': 2}
    for data_sample in data_iter:
        tem = data_sample[language_index[language]]
        if language == SBT_LANGUAGE:
            sbts,idss = sbt.start(3,tem)
            sbts = [s.split("_") for s in sbts]
            sbts = [item for sublist in sbts for item in sublist]
            yield sbts
            continue
        if language == IDS_LANGUAGE:
            sbts,idss = sbt.start(3,tem)
            yield (sbts,idss)
            continue
        if language == 'spe':
            sbts,idss = sbt.start(3,tem)
            sbts2 = [s.split("_")[0] for s in sbts]
            yield (sbts,sbts2)
            continue
        # Replace strings and numbers
        if language == CODE_LANGUAGE:
            tem = replaceSaI(tem)
        # Split words by _ symbols
        tem = [s.split("_") for s in tem]
        tem = [item for sublist in tem for item in sublist]
        # split spaces
        tem = [s.split(" ") for s in tem]
        tem = [item for sublist in tem for item in sublist]
        # Split CamelCase
        if language == CODE_LANGUAGE:
            split_camel_case(tem)
            tem = [item for sublist in tem for item in sublist]
        tem = [word.lower() for word in tem]
        yield tem

# 设置特殊符号和序列
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<start>', '<end>', '<str>','<int>']

for ln in [SBT_LANGUAGE,NL_LANGUAGE]:
    # 数据迭代器
    train_iter = MapperIterDataPipe(split='train')
    # 构建词汇表
    vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(train_iter, ln),
                                                    min_freq=1,
                                                    specials=special_symbols,
                                                    max_tokens=30000,
                                                    special_first=True)
for ln in [CODE_LANGUAGE]:
    # 数据迭代器
    train_iter = MapperIterDataPipe(split='train')
    # 构建词汇表
    vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(train_iter, ln),
                                                    min_freq=1,
                                                    specials=special_symbols,
                                                    max_tokens=50000,
                                                    special_first=True)


# 处理默认值
for ln in [CODE_LANGUAGE, NL_LANGUAGE,SBT_LANGUAGE]:
  vocab_transform[ln].set_default_index(UNK_IDX)

# 词汇表大小
SRC_VOCAB_SIZE = len(vocab_transform[CODE_LANGUAGE])
TGT_VOCAB_SIZE = len(vocab_transform[NL_LANGUAGE])

# 生成全流程函数
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            #print(txt_input)
            txt_input = transform(txt_input)
        return txt_input
    return func

# 拼接BOS/EOS
def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))

text_transform = {}
for ln in [CODE_LANGUAGE, NL_LANGUAGE]:
    text_transform[ln] = sequential_transforms(vocab_transform[ln], #数值化
                                               tensor_transform) # 添加 BOS/EOS 创建 tensor

#print(text_transform[NL_LANGUAGE]("A elderly man holds a can behind his back as he strolls by a beautiful flower market."))

# 数据批处理
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform[CODE_LANGUAGE](src_sample))
        tgt_batch.append(text_transform[NL_LANGUAGE](tgt_sample))
    # 填充值
    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch

mvocab = vocab_transform[CODE_LANGUAGE].get_itos()

print(mvocab[6000])
print(mvocab[6100])
print(mvocab[6200])
print(mvocab[6300])
print(mvocab[6400])
print(mvocab[6500])
print(mvocab[6600])
print(mvocab[6700])
print(mvocab[6800])
print(mvocab[6900])




'''
for code, nl in train_dataloader:
    print('code')
    print(code.shape)
    print('nl')
    print(nl.shape)
    break
'''