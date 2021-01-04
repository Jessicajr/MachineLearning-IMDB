from torchtext import data
from torchtext.vocab import Vectors
from torch.nn import init
from tqdm import tqdm
# import spacy
import  sys
import tqdm
import torch.nn as nn
from torchtext.data import Iterator, BucketIterator

# spacy_en = spacy.load('en')

def tokenizer(text): # create a tokenizer function
    return (' '.join(text.split())).split()

def loadData(path_train,path_valid,path_predict):

    TEXT = data.Field(tokenize=tokenizer,batch_first=True,
                      init_token='<SOS>', eos_token='<EOS>',lower=True)
    LABEL = data.Field(sequential=False, use_vocab=False)
    INDEX = data.Field(sequential=False,use_vocab=False)
    tv_datafields = [("id",INDEX),
                     ("review", TEXT), ("sentiment", LABEL)]
    train,valid,test= data.TabularDataset.splits(path='./public_data/',train='train_s.csv',
            validation='test_s.csv',test='new_test_data.csv', format='csv',fields=tv_datafields,skip_header=True)
    # test = data.TabularDataset.splits(path='./public_data/',test='test_data.csv',
    #            format='csv',
    #            skip_header=True,
    #            fields=[("id",INDEX),
    #                  ("review", TEXT)])
    TEXT.build_vocab(train, vectors="glove.6B.300d")
    vocab = TEXT.vocab
    # embed = nn.Embedding(len(vocab), 100)
    # embed.weight.data.copy_(vocab.vectors)
    train_iter, val_iter = BucketIterator.splits((train, valid),
                                                 # 我们把Iterator希望抽取的Dataset传递进去
                                                 batch_sizes=(16, 16),
                                                 device=-1,
                                                 # 如果要用GPU，这里指定GPU的编号
                                                 sort_key=lambda x: len(x.review),
                                                 # BucketIterator 依据什么对数据分组
                                                 sort_within_batch=False,
                                                 repeat=False)
    test_iter = Iterator(test, batch_size=16,
                     device=-1,
                     sort=False,
                     sort_within_batch=False,
                     repeat=False)

    return vocab,train_iter,val_iter,test_iter

if __name__ == '__main__':
    vocab, train_iter, val_iter, test_iter = loadData(path_train='public_data/train_s.csv',
                                                      path_valid='public_data/test_s.csv', path_predict=
                                                      'public_data/test_data.csv')
    for item in val_iter:
        print(item)