import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys

class BiLSTM(nn.Module):
    def __init__(self, emb_size, hidden_size, n_layers, dropout=0.1,cuda=False):
        super(BiLSTM,self).__init__()
        self.lstm = nn.LSTM(input_size=emb_size, hidden_size=hidden_size, num_layers=n_layers, dropout=dropout,bidirectional=True)
        self.use_cuda = cuda



    def bi_fetch(self, rnn_outs, batch_size, max_len):
        rnn_outs = rnn_outs.view(batch_size, max_len, 2, -1)

        # (batch_size, max_len, 1, -1)
        fw_out = torch.index_select(rnn_outs, 2, Variable(torch.LongTensor([0])))
        bw_out = torch.index_select(rnn_outs, 2, Variable(torch.LongTensor([1])))

        fw_out = fw_out.squeeze(2)   # (batch_size, max_len, hid)
        bw_out = bw_out.squeeze(2)
        bw_out = bw_out.flip(1)
        outs = torch.add(fw_out, bw_out)
        return outs


    def forward(self, x):
        '''

        :param x: (batch_size*seq_length*emb_size)
        :return: (batch,max_len,hidden),(batch,2*hidden)
        '''
        batch_size = x.shape[0]
        x_ = x.transpose(0,1).contiguous()
        #print(x_.dtype)
        #print(x_.shape)
        output, (hid, _) = self.lstm(x_)  #(len_seq,batch_size,n_hidden * 2)
        output = output.transpose(0,1).contiguous()         #(batch_size,len_seq,n_hidden * 2)
        #print(output.size())
        output = self.bi_fetch(output, output.shape[0], output.shape[1])
        #print(hid.size())
        hid_out = torch.cat([hid[-1, :, :], hid[-2, :, :]],dim=-1)
        #f_output,b_output = output.chunk()

        return output,hid_out

class CNNClassifier(nn.Module):
    def __init__(self, embedding_size, kernel_out=100, kernel_sizes=(2,3,4), dropout=0.5):
        super(CNNClassifier,self).__init__()
        self.convs = nn.ModuleList([nn.Conv2d(1,kernel_out,(K,embedding_size)) for K in kernel_sizes])
        self.dropout = dropout

    def forward(self, x):
        '''

        :param x: (batch,seq_length,emb_size/lstm_hid_size)
        :return:(batch,kernel_out*Ks)
        '''
        x = x.unsqueeze(1)  #b,1,len,emb
        try:
            conv_out = [F.relu(conv(x)).squeeze(3) for conv in self.convs]   #[batch, kernel_out, seq_length] *ks
            pool_out = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in conv_out]   #[(batch,kernel_out)]
        except:
            print(x.shape)
            padding = nn.ConstantPad2d(padding=(0, 0, 4-x.shape[2], 0), value=0)
            x = padding(x)
            conv_out = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [batch, kernel_out, seq_length] *ks
            pool_out = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in conv_out]  # [(batch,kernel_out)]
        out = torch.cat(pool_out, 1)
        return out

class FeedFroward(nn.Module):
    def __init__(self, n_class, hidden_size, dropout, input_size):
        super(FeedFroward, self).__init__()
        self.dropout = nn.Dropout(dropout)
        #self.linear1 = nn.Linear(input_size,hidden_size)
        self.linear2 = nn.Linear(input_size, n_class)
        #self.relu = nn.

    def forward(self, x):
        '''

        :param x: batch,
        :return:
        '''
        #print(x)
        #out1 = self.tanh(self.linear1(x))
        out = self.dropout(x)
        out = self.linear2(out)
        return out





