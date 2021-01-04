import torch
import random
import os
import sys
import csv
from model.dataloader import *
from model.text import myVocab
from model.Trainer import Trainer
from model.TotalModules import RCModule
from torch.utils.data import DataLoader
from model.newDataLoader import *

def main():
    vocab, train_iter, val_iter, test_iter = loadData(path_train='public_data/train_s.csv',
                                                      path_valid='public_data/test_s.csv', path_predict=
                                                      'public_data/test_data.csv')

    # for item1,item2 in zip(train_iter,test_iter):
    #     print(item1)
    #     print(item2)
    #     sys.exit()
    device = torch.device('cpu')
    model = RCModule(n_layers=2,
                     embeddings_size=300,
                     vocab=vocab,
                     padding_idx=0,
                     rnn_dropout=0.5,
                     cnn_dropout=0.5,
                     ff_dropout=0.5,
                     n_class=2,
                     pretrained=False)

    model_trainer = Trainer(model, train_data=None, predict_dataset=test_iter, test_dataset=None, batch_size=8,vocab=vocab)
    path = './ckpt/rnn_cnn_step1'
    state_dict = torch.load(path, map_location=device)
    model_trainer.load_state_dict(state_dict)
    with torch.no_grad():
        ret_list,indexs = model_trainer.predict(vocab)
    return ret_list,indexs


if __name__ =="__main__":
    #main()
    fw = open('predict_rnn_cnn_1.csv', 'w', newline="", encoding='utf-8')
    csv_write = csv.writer(fw)
    csv_write.writerow(['', 'sentiment'])
    label,indexs = main()
    for ind,lab in zip(indexs,label):
        if lab==1:
            sent = 'positive'
        else:
            sent = 'negative'
        csv_write.writerow([str(ind),sent])
    fw.close()



