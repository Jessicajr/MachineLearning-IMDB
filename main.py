# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import torch
import math
import torch
import csv
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report,  precision_recall_fscore_support,precision_score


def flip(x, dim):
    xsize = x.size()
    print(xsize)
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    print(x)
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1) - 1,
                                                                 -1, -1), ('cpu')[x])().long(), :]
    return x.view(xsize)


def print_hi(path):
    # Use a breakpoint in the code line below to debug your script.
    fr_data = open(path,encoding='utf-8') # Press Ctrl+F8 to toggle the breakpoint.
    reader = csv.reader(fr_data)
    s = set()
    for i in reader:
        s.add(i[1])
    for item in s:
        print(item)
    print(len(s))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    '''
    import torch.nn as nn
    lis=[]
    lis1=[]
    m = nn.LogSoftmax()
    loss = nn.NLLLoss()
    # input is of size nBatch x nClasses = 3 x 5
    input = torch.autograd.Variable(torch.randn(3, 5), requires_grad=True)
    print(input)
    # each element in target has to have 0 <= value < nclasses
    target = torch.autograd.Variable(torch.LongTensor([1, 0, 4]))
    input_= m(input)
    print(input_)
    #print(torch.argmax(input_,dim=-1).data)
    lis+=torch.argmax(input_,dim=-1).data
    lis1+=target
    output = loss(input_, target)
    print(output)
    print(precision_score(lis, lis1, average = "macro"))
    '''