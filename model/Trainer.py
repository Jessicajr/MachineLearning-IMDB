import torch
import torch.nn as nn
from model.text import myVocab
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy
from sklearn.metrics import accuracy_score,f1_score, precision_score, recall_score, classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm
from .utils import pad_sequence
import json
import logging
import sys
logger = logging.getLogger('enn-cnn')
logger.setLevel(logging.INFO)
fh = logging.FileHandler('rnnAndcnn.log', encoding='utf-8')
fh.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
# 记录一条日志
logger.info('python logging test')

class Trainer:
    def __init__(self, model, train_data, vocab, predict_dataset=None, test_dataset=None, batch_size=8,batch_split=1):
        self.model = model
        self.vocab = vocab
        #self.loss = nn.NLLLoss()
        self.loss = nn.CrossEntropyLoss()
        #self.loss = nn.BCELoss()
        self.batch_split = batch_split
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        if train_data is not None:
            self.train_dataloader = train_data
        if test_dataset is not None:
            self.test_dataloader = test_dataset
        if predict_dataset is not None:
            self.predict_dataloader = predict_dataset
        # if train_data is not None:
        #     self.train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False, collate_fn=self.collate_func)
        # if test_dataset is not None:
        #     self.test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=self.collate_func)
        # if predict_dataset is not None:
        #     self.predict_dataloader = DataLoader(predict_dataset,batch_size=batch_size,shuffle=False, collate_fn=self.collate_func)


    # def collate_func(self, data):
    #     #print(data)
    #     passage, y = zip(*data)
    #     contexts = []
    #     if max(map(len, passage)) > 0:
    #         h = [torch.tensor(d, dtype=torch.float) for d in passage]
    #         h = pad_sequence(h, batch_first=True, padding_value=self.model.padding_idx)
    #         #print(h)
    #         contexts.append(h)
    #     #y = [torch.tensor(d, dtype=torch.long) for d in y]
    #     #y = pad_sequence(y, batch_first=True, padding_value=self.model.padding_idx)
    #     y = torch.tensor(y, dtype=torch.long)
    #     return contexts, y

    def state_dict(self):
        return {'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict()}

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict['model'], strict=False)
        self.optimizer.load_state_dict(state_dict['optimizer'])


    def _eval_train(self, epoch):
        self.model.train()

        tqdm_data = tqdm(self.train_dataloader, desc='Train (epoch #{})'.format(epoch))
        loss = 0
        # for i, (context, targets) in enumerate(tqdm_data):
        for i,item in enumerate(self.train_dataloader):
            context = item.review
            targets = item.sentiment
            self.optimizer.zero_grad()
            predict = self.model(context)
            # print('predict:{}'.format(predict))
            # print('targets:{}'.format(targets))
            # predict = torch.sigmoid(model_out)
            # print(predict,targets)
            batch_loss = self.loss(predict, targets)
            #print(batch_loss)
            batch_loss.backward()
            # if (i+1)% self.batch_split == 0:
            self.optimizer.step()
                # self.optimizer.zero_grad()
            loss = (i*loss + batch_loss.item())/(i+1)
            # if i%500==0:
            #     print('i:{},loss:{}'.format(i,loss))
            tqdm_data.set_postfix({'loss': loss,'i': i})
        log_dict = {'epoch': epoch, 'loss': loss}
        log_dict_json = json.dumps(log_dict, ensure_ascii=False)
        logger.info(log_dict_json)

    def _eval_test(self):
        self.model.eval()
        # tqdm_data = tqdm(self.test_dataloader, desc='Test')
        loss = 0
        predicts = []
        labels = []
        with torch.no_grad():
            for i, item in enumerate(self.test_dataloader):
                passage = item.review
                y = item.sentiment
                predict = self.model(passage)
                # print(predict)
                predict = torch.argmax(predict, dim=-1)
                # predict = torch.sigmoid(model_out)
                # predict_lis_ = predict.data.numpy().tolist()
                # predict_lis = [0 if item < 0.5 else 1 for item in predict_lis_]
                predicts.extend(predict.numpy().tolist())
                labels.extend(y.data.numpy().tolist())
            a = accuracy_score(labels, predicts)
            t = classification_report(labels, predicts)
            print(t)
            log_dict = {'accuracy': a}
            log_dict_json = json.dumps(log_dict, ensure_ascii=False)
            logger.info(log_dict_json)

    def predict(self,vocab):
        # tqdm_data = tqdm(self.predict_dataloader, desc="Predict")
        predicts = []
        with torch.no_grad():
            idxs = []
            for i, item in enumerate(self.predict_dataloader):
                # print(item.fields)
                passage = item.review
                idx = item.id
                predict = self.model(passage)
                predict = torch.argmax(predict, dim=-1)
                predicts.extend(predict.numpy().tolist())
                idxs.extend(idx.numpy().tolist())
        return predicts,idxs


    def train(self,start_epoch, epochs,after_epoch_funcs=[]):
        for epoch in range(start_epoch, epochs):
            self._eval_train(epoch)
            self.test()
            for func in after_epoch_funcs:
                func(epoch)

    def test(self):
        if hasattr(self, 'test_dataloader'):
            self._eval_test()



