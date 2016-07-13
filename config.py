class Config(object):
    data_path = '/home/ankit/devel/data/MIMIC3pk'
    init_scale = 0.05
    learning_rate = 1e-3
    max_grad_norm = 5
    num_layers = 2
    num_steps = 35
    hidden_size = 650
    learn_emb_size = 150
    max_epoch = 6
    max_max_epoch = 39
    keep_prob = 0.5
    batch_size = 20
    vocab_size = 25000
    print_every = 100
    pretrained_emb = True
