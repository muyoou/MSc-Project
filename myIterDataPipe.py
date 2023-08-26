import ast
from torchdata.datapipes.iter import IterDataPipe
import json
import re

class MapperIterDataPipe(IterDataPipe):
    def __init__(self, split,file = None) -> None:
        super().__init__()
        self.codeArr = []
        self.nlArr = []
        if split=='train':
            self.dir = []
            if file == None:
                self.dir.append("data3/"+split+"3.jsonl")
        elif  split == 'tv':
            self.dir = []
            self.dir.append("data3/train3.jsonl")
            self.dir.append("data3/valid2.jsonl")
        else:
            self.dir = ["data3/"+split+"2.jsonl"]

    def __iter__(self):
        for filedir in self.dir:
            with open(filedir, 'r') as file:
                for line in file:
                    #if len(json.loads(line)['code_tokens'])>100: continue
                    #if len(json.loads(line)['docstring_tokens'])>20: continue
                    #if self.contains_chinese_characters(json.loads(line)['original_string']): continue
                    try: ast.parse(json.loads(line)['original_string'])
                    except:continue
                    #if json.loads(line)['original_string'] in self.codeArr: continue
                    #if json.loads(line)['docstring'] in self.nlArr : continue
                    #self.codeArr.append(json.loads(line)['original_string'])
                    #self.nlArr.append(json.loads(line)['docstring'])
                    yield (json.loads(line)['code_tokens'],json.loads(line)['docstring_tokens'],json.loads(line)['original_string'])

    def __len__(self):
        lennum = 0
        for filedir in self.dir:
            with open(filedir, 'r') as file:
                lennum +=len(list(file))
        return lennum
    
    def contains_chinese_characters(self,input_str):
        chinese_pattern = re.compile("[\u4e00-\u9fa5]")
        match = chinese_pattern.search(input_str)
        return match is not None