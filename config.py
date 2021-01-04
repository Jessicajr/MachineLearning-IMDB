from attrdict import AttrDict


# transformer config
def get_model_config_context():
    default_config = openai_transformer_config()
    config = AttrDict({'vocab_path': './parameters/vocab.txt',
                       'n_layers': default_config.n_layers,
                       'n_pos_embeddings': 512,
                       'embeddings_size': default_config.embeddings_size,
                       'n_heads': default_config.n_heads,
                       'dropout': default_config.dropout,
                       'embed_dropout': default_config.embed_dropout,
                       'attn_dropout': default_config.attn_dropout,
                       'ff_dropout': default_config.ff_dropout,
                       'max_seq_len': 300,
                       'beam_size': 2,
                       'diversity_coef': 0,
                       'diversity_groups': 1,
                       'annealing_topk': 20,
                       'temperature': 0.8,
                       'annealing': 0,
                       'length_penalty': 5,
                       'n_segments': None,
                        'hidden_size': 128})

    return config

def get_trainer_config_context():
    config = AttrDict({'n_epochs': 500,
                       'batch_size': 128,
                       'batch_split': 64,
                       'lr': 6.25e-5,
                       'lr_warmup': 1000,
                       'lm_weight': 0.5,
                       'risk_weight': 0,
                       'n_jobs': 4,
                       'label_smoothing': 0.1,
                       'clip_grad': 1.0,
                       'test_period': 1,
                       'seed': 0,
                       'device': 'cuda',
                       'load_last':False,
                       'openai_parameters_dir': '/output/workspace/KnowledgeEncode/parameters/chinese_pretrain.pt',
                       'last_checkpoint_path': '/output/workspace/KnowledgeEncode/checkpoints/last_checkpoint_kn',
                       'interrupt_checkpoint_path': '/output/workspace/KnowledgeEncode/checkpoints/interrupt_checkpoint',
                       'train_dataset': '/output/workspace/KnowledgeEncode/data/train_data.json',
                       'test_dataset': '/output/workspace/KnowledgeEncode/data/valid_data.json'})

    return config


def get_test_config_context():
    config = AttrDict({'seed': 0,
                       'device': 'cuda',
                       'last_checkpoint_path': './ensemble.pt'})

    return config

def get_test_config_context_ensemble():
    config = AttrDict({'seed': 0,
                       'device': 'cuda',
                       'last_checkpoint_path': ['./data/baidu_best_checkpoint_479.pt']})
    return config
