import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import *

class RCModule(nn.Module):
    def __init__(self, n_layers, embeddings_size, vocab,
                 padding_idx, rnn_dropout, cnn_dropout, ff_dropout, rnn_hidden_size=128, ff_hidden_size=128, cnn_out=100, n_class=2,kernel_sizes=(2,3,4),pretrained = False):
        super(RCModule, self).__init__()
        self.padding_idx = padding_idx
        self.embeddings_size = embeddings_size
        self.ff_hidden_size = ff_hidden_size

        self.rnn_hidden_size = rnn_hidden_size
        self.n_layers = n_layers
        self.rnn_dropout = rnn_dropout
        self.cnn_dropout = cnn_dropout
        self.ff_dropout = ff_dropout
        self.cnn_out = cnn_out
        if not pretrained:
            self.embedding = nn.Embedding(len(vocab), embeddings_size)
            self.embedding.weight.data.copy_(vocab.vectors)
        else:
            self.embedding = None

        self.rnnModel = BiLSTM(self.embeddings_size, rnn_hidden_size, n_layers, rnn_dropout)
        self.cnnModel = CNNClassifier(embedding_size=rnn_hidden_size, kernel_out=cnn_out, kernel_sizes=kernel_sizes,  dropout=cnn_dropout)
        self.rnnFF = FeedFroward(n_class, ff_hidden_size, ff_dropout, 2*rnn_hidden_size)
        self.cnnFF = FeedFroward(n_class, ff_hidden_size, ff_dropout, len(kernel_sizes)*cnn_out)
        #self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        #print(x)
        if self.embedding is not None:
            x_enc = self.embedding(x)
        else:
            #print(x)
            x_enc = x
        rnn_out, hid_out = self.rnnModel(x_enc)
        # print(rnn_out.size(),hid_out.size())
        rnn_ff_out = self.rnnFF(hid_out)    #batch * n_class
        cnn_out = self.cnnModel(rnn_out)
        cnn_ff_out = self.cnnFF(cnn_out)
        # out = cnn_ff_out
        out = (cnn_ff_out+rnn_ff_out)/2
        return out











