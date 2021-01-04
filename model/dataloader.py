import torch
from torch.utils.data import Dataset
from model.text import myVocab
from collections import defaultdict
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import re
import csv
import random
import sys
import json
import nltk
import numpy as np


def writeFile(file, dict, lines):
    csv_write = csv.writer(file)
    index = 0
    #tag_nums = len(dict.keys())
    while index < lines:
        seed = 0
        if(len(dict)>1):
            seed = random.randint(0, len(dict)-1)
        try:
            tag, content = dict[seed].pop()
        except:
            print(dict)
            print(index)
            sys.exit()
        if len(dict[seed]) == 0:
            del dict[seed]
        index += 1
        csv_write.writerow([str(index), tag, content])



def readFile(org_path,train_path,test_path):
    fr_data = open(org_path, 'r', encoding='utf-8')
    train = open(train_path, 'w', newline="", encoding='utf-8')
    test = open(test_path, 'w', newline="", encoding='utf-8')
    csv_reader = csv.reader(fr_data)
    dict_train =[[] for i in range(10)]
    dict_test = [[] for i in range(10)]
    map = {'negative': 0, 'positive': 1}
    for (i, item) in enumerate(csv_reader):
        if(i == 0):
            continue
        else:
            tag = map[item[1]]
            content = item[2]
            if(len(dict_train[tag]) >= 700):
                dict_test[tag].append((item[1], content))
            else:
                dict_train[tag].append((item[1], content))
    count = 0
    for i in range(10):
        count += len(dict_train[i])
    print(count)
    #print(dict_train)
    #print(dict_test)
    writeFile(train, dict_train, 7000)
    writeFile(test, dict_test, 3000)
    fr_data.close()
    train.close()
    test.close()

class BertEmbedding():
    def __init__(self, path):
        self.tokenizer = BertTokenizer.from_pretrained(path)
        self.model = BertModel.from_pretrained(path)

    # def

    def embedding(self, text):
        ret=[]
        eos_list = ['.', '?', '!']
        text = ' '.join(text.split())
        lis_text = list(text)
        lis = [[]]
        index = 0
        for item in lis_text:
            lis[index].append(item)
            if item in eos_list:
                lis[index].append('[SEP]')
                lis.append([])
                index += 1
        if len(lis[-1]) != 0:
            if lis[-1][-1] != '[SEP]':
                lis[-1].append('[SEP]')
        for item in lis:
            if len(item)==0:
                continue
            sub_text = '[CLS]'+''.join(item)
            print(sub_text)
            tokenized_text = self.tokenizer.tokenize(sub_text)
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
            tokens_tensor = torch.tensor([indexed_tokens])
            segments_ids = [1] * len(tokenized_text)
            segments_tensors = torch.tensor([segments_ids])
            with torch.no_grad():
                encoded_layers, pooled_output = self.model(tokens_tensor, segments_tensors)
                #print(pooled_output.shape)
                ret.append((pooled_output.numpy().tolist())[0])
                #print((pooled_output.numpy().tolist())[0])
            #print(len(ret))
        return ret

class embeddingDataset(Dataset):
    def __init__(self, path, bert_path, max_lengths=2048, flag='train', write_path = '../data/train_emb.json',write = True):
        if write:
            embedding_f = open(write_path,'w')
        else:
            embedding_f = open(write_path,'r')
            #embedding_f2 = open('data/emb_test.json','r')
        self.max_lengths = max_lengths
        bert = BertEmbedding(bert_path)
        dict_y = {'negative': 0, 'positive': 1}
        if write:
            self.data = self.make_dataset(dict_y, path, flag, bert)
            #print(type(self.data))
            json.dump(self.data, embedding_f)
        else:
            self.data = json.load(embedding_f)
        embedding_f.close()

    @staticmethod
    def make_dataset(dict_y, path, flag, embedder):
        dataset = []
        fr_data = open(path, encoding='utf-8')
        reader = csv.reader(fr_data)

        line1 = 0
        for i, item in enumerate(reader):
            if i==0:
                continue
            y = -1
            if (flag == 'train'):
                #print(dict_y.keys())
                y = dict_y[item[2]]
                passage = embedder.embedding(item[1][:512])
                print(passage)
                sys.exit()
            else:
                passage = embedder.embedding(item[1][:512])
            if (i % 100)==0:
                print(i)
            dataset.append([passage, y])
        #print(dataset[0][0])
        print('finish embedding')
        return dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        passage, y = sample
        return passage,y

def calculateDataCount(org_path):
    f_r = open(org_path,'r',encoding='utf-8')
    csv_r = csv.reader(f_r)
    count_positive = 0
    count_negatiev =0
    for i,item in enumerate(csv_r):
        if i==0:
            continue
        if item[2]=='positive':
            count_positive+=1
        if item[2] == 'negative':
            count_negatiev+=1
    print(count_positive,count_negatiev)
    f_r.close()

def splitData(org_path,f_train,f_test):
    f_r = open(org_path, 'r', encoding='utf-8')
    csv_r = csv.reader(f_r)
    fw_train = open(f_train,'w',encoding='utf-8',newline="")
    fw_test = open(f_test,'w',encoding='utf-8',newline="")
    csv_train = csv.writer(fw_train)
    csv_test = csv.writer(fw_test)
    csv_train.writerow(['index','review','sentiment'])
    csv_test.writerow(['index', 'review', 'sentiment'])
    for i,item in enumerate(csv_r):
        if i==0:
            continue
        seed = random.randint(1,10)
        if seed%5==0:
            csv_test.writerow([item[i] for i in range(3)])
        else:
            csv_train.writerow([item[i] for i in range(3)])
    f_r.close()
    fw_train.close()
    fw_test.close()

if __name__=="__main__":
    train_dataset = embeddingDataset('../public_data/s_train.csv', '../bert_base_uncased', write_path='../data/s_train1.json',
                                     write=True)
    #tokenizer = BertTokenizer.from_pretrained('../bert-base-uncased')
    # model = BertModel.from_pretrained('../bert-base-uncased')
    # test_dataset = embeddingDataset('../public_data/s_test.csv', '../bert-base-uncased', write_path='../data/s_test.json',
    #                                 write=True)



    '''
    fr_data = open("../data/labeled_data.csv", 'a+', newline="", encoding='utf-8')
    #csv_reader = csv.reader(fr_data)
    #count=0
    #for(i, item) in enumerate(csv_reader):
        #print(item[2])
    #    count += 1
    #    if(i>7000):
    #        print(item[0])
    #print(count)


    csv_write = csv.writer(fr_data)
    index = 9000
    start = 131604
    end = 224235
    index_set = set()
    for i in range(1000):
        while True:
            seed = random.randint(start, end)
            if seed not in index_set:
                index_set.add(seed)
                break
        f = open("../data/娱乐/"+str(seed)+".txt",'r',encoding='utf-8')
        content = []
        for line in f:
            content.append(line.strip())
        content = '\t'.join(content)
        csv_write.writerow([str(index+i), "娱乐", content])
        f.close()

    fr_data.close()
    '''

    #readFile("../data/labeled_data.csv", "../data/train.csv", "../data/test.csv")

    '''
    vocab = myVocab("../vocab.txt")
    dataset = charDataset("../data/train.csv",vocab)
    print(dataset.__len__())

    for i in range(10):
        passage,y = dataset.__getitem__(i)
        print(passage)
        print('passage:{}\ny:{}\n'.format(vocab.ids2string(passage),y))
    #print(dataset.__getitem__(0))
    '''
    nltk.download()