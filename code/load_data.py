import json
import torch
from torch.utils.data import TensorDataset
import numpy as np
from transformers import RobertaTokenizer

class Dictionary(object):

    def __init__(self):
        self.word2tkn = {'':0}
        self.tkn2word = ['']

    def add_word(self,word):
        if word not in self.word2tkn:
            self.tkn2word.append(word)
            self.word2tkn[word] = len(self.tkn2word) - 1
        return self.word2tkn[word]

class Corpus(object):

    def __init__(self,max_sent_len):
        self.dictionary = Dictionary()
        self.max_sent_len = max_sent_len
        self.train = self.tokenize("dataset/implicit_train.json")
        self.test = self.tokenize("dataset/implicit_test.json")
        self.vaild = self.tokenize("dataset/implicit_dev.json")

    def tokenize(self,path):
        #tokenizer = RobertaTokenizer.from_pretrained('model\\roberta-base\\tokenizer.json')
        vocab_file = 'model/vocab.json'
        merges_file = 'model/merges.txt'
        tokenizer = RobertaTokenizer(vocab_file, merges_file)
        with open(path, 'r', encoding='utf-8') as fp:

            idss = []
            labels = []
            masks = []
            data = json.load(fp)  # 读取一条数据
            for one_data in data:


                arg1 = one_data['arg1']
                arg2 = one_data['arg2']
                label = one_data['label']
                conn = one_data['conn']
                #用roberta的tokenizer的encode，会返回两个参数
                ids = tokenizer.encode_plus(arg1,arg2,
                                            max_length=self.max_sent_len,
                                            pad_to_max_length = True)
                #会返回一个input_ids,一个attention_mask，分别储存
                idss.append(ids['input_ids'])
                masks.append(ids['attention_mask'])
                #因为有的标签有两个，有的标签只有一个，所以统一为1*2，方便使用，如果只有一个标签就
                #多append一个不存在的标签5，扩充到长度为2
                if len(label) == 1:
                    label.append(5)
                labels.append(label)
            idss = torch.tensor(np.array(idss))
            labels = torch.tensor(np.array(labels))
            masks = torch.tensor(np.array(masks)).float()


        return TensorDataset(idss,masks,labels)