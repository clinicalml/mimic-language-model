class Config(object):
    data_path = '/home/ankit/devel/data/MIMIC3pk'
    init_scale = 0.05
    learning_rate = 1e-3
    max_grad_norm = 5
    num_layers = 2
    num_steps = 15
    hidden_size = 650
    learn_emb_size = 150
    max_epoch = 6
    keep_prob = 0.5
    batch_size = 10
    print_every = 100
    pretrained_emb = True
