import preprocessing as pre
from myIterDataPipe import MapperIterDataPipe

vocabList = {}
file = None

def vocabulary(lang):
    output_file_path = "data2/vocabulary/"+lang
    vocab = pre.vocab_transform[lang].get_itos()
    vocabList[lang] = []
    print(vocab[17565])
    with open(output_file_path, "w", encoding="utf-8") as f:
        for item in vocab:
            f.write(item + "\n")
#vocabulary()

def chars (model,lang):
    output_file_path = "data2/"+model+"/"+lang+".char"
    train_iter = MapperIterDataPipe(split=model,file=file)
    with open(output_file_path, "w", encoding="utf-8") as f:
        for item in pre.yield_tokens(train_iter, lang):
            f.write(" ".join(item) + "\n")
#chars('valid')

def embedding (model,lang):
    vocab = pre.vocab_transform[lang]
    output_file_path = "data2/"+model+"/"+lang
    train_iter = MapperIterDataPipe(split=model,file=file)
    index = 0
    with open(output_file_path, "w", encoding="utf-8") as f:
        for item in pre.yield_tokens(train_iter, lang):
            if index == 1: print(item)
            newItem = vocab.lookup_indices(item)
            if index == 1: print(newItem)
            newItem = [str(i) for i in newItem]
            if index == 1: print(newItem)
            index+=1
            f.write(" ".join(newItem) + "\n")
#embedding('train')

def ids(model):
    for lang in [pre.IDS_LANGUAGE]:
        output_file_path = "data2/"+model+"/"+lang
        train_iter = MapperIterDataPipe(split=model,file=file)
        with open(output_file_path, "w", encoding="utf-8") as f:
            for _,item in pre.yield_tokens(train_iter, lang):
                item = [str(i) for i in item]
                f.write(" ".join(item) + "\n")
#ids('valid')

def sbt (model):
    for lang in [pre.SBT_LANGUAGE]:
        vocab = pre.vocab_transform[lang]
        output_file_path = "data2/"+model+"/"+lang
        train_iter = MapperIterDataPipe(split=model,file=file)
        with open(output_file_path, "w", encoding="utf-8") as f:
            for item,item2 in pre.yield_tokens(train_iter, 'spe'):
                newItem = vocab.lookup_indices(item)
                newItem2 = vocab.lookup_indices(item2)
                for index in range(len(newItem)):
                    if newItem[index] == 0 and newItem2[index] != 0:
                        newItem[index] = newItem2[index]
                newItem = [str(i) for i in newItem]
                f.write(" ".join(newItem) + "\n")

#sbt('valid')


chars('valid',pre.CODE_LANGUAGE)
chars('train',pre.NL_LANGUAGE)
chars('valid',pre.NL_LANGUAGE)
vocabulary(pre.CODE_LANGUAGE)
vocabulary(pre.NL_LANGUAGE)
vocabulary(pre.SBT_LANGUAGE)
embedding('train',pre.CODE_LANGUAGE)
embedding('train',pre.NL_LANGUAGE)
embedding('valid',pre.CODE_LANGUAGE)
embedding('valid',pre.NL_LANGUAGE)
ids('train')
ids('valid')
sbt('train')
sbt('valid')
