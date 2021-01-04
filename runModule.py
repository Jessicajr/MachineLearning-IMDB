import torch
import random
import os
import sys
import json
from model.dataloader import embeddingDataset
from model.newDataLoader import loadData
from model.text import myVocab
from model.Trainer import Trainer
from model.TotalModules import RCModule
from torch.utils.data import DataLoader

def main():
    vocab, train_iter, val_iter, test_iter = loadData(path_train='public_data/train_s.csv',path_valid='public_data/test_s.csv',path_predict=
                                                      'public_data/test_data.csv')
    load_last = False
    test = True
    last_ckpt_path = './ckpt/rnn_cnn_step'
    device = torch.device('cpu')
    n_epoch = 500
    model = RCModule(n_layers=2,
                     embeddings_size=300,
                     vocab= vocab,
                     padding_idx=0,
                     rnn_dropout=0.5,
                     cnn_dropout=0.5,
                     ff_dropout=0.5,
                     n_class=2,
                     pretrained=False)
    # train_dataset = embeddingDataset('./public_data/s_train.csv', 'bert_base_uncased',write_path='data/s_train.json', write=False)
    # test_dataset = embeddingDataset('./public_data/s_test.csv', 'bert_base_uncased',write_path='data/s_test.json', write=False)
    model_trainer = Trainer(model, train_data=train_iter, predict_dataset=None, test_dataset=val_iter, batch_size=16, vocab=vocab,batch_split=1)
    start_epoch = 0
    init_epoch = 1



    if load_last:
        state_dict = torch.load(last_ckpt_path + str(init_epoch - 1), map_location=device)
        model_trainer.load_state_dict(state_dict)
        # start_epoch = int(cop.sub('', trainer_config.last_checkpoint_path.split('/')[-1])) + 1
        start_epoch = init_epoch
        print('Weights loaded from {}'.format(last_ckpt_path + str(init_epoch - 1)))

    def save_func(epoch):
        torch.save(model_trainer.state_dict(), last_ckpt_path)
        torch.save(model_trainer.state_dict(), last_ckpt_path + str(epoch))
        if os.path.exists(last_ckpt_path + str(epoch - 100)):
            os.remove(last_ckpt_path + str(epoch - 100))


    try:
        model_trainer.train(start_epoch, n_epoch, after_epoch_funcs=[save_func])
    except (KeyboardInterrupt, Exception, RuntimeError) as e:
        torch.save(model_trainer.state_dict(), './interrupt')
        raise e

    if test and val_iter is not None:
        try:
            model_trainer.test()
        except(KeyboardInterrupt, Exception, RuntimeError) as e:
            raise e

if __name__=="__main__":
    main()

