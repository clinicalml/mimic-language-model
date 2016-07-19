class Config(object):
    data_path = '/home/ankit/devel/data/MIMIC3pk'
    init_scale = 0.05
    learning_rate = 1e-3
    max_grad_norm = 5
    num_layers = 2
    num_steps = 15
    hidden_size = 650
    learn_wordemb_size = 150
    max_epoch = 6
    keep_prob = 0.5
    batch_size = 20
    print_every = 200
    pretrained_emb = True
    conditional = True
    fixed_len_features = set(['gender', 'has_dod', 'has_icu_stay', 'admission_type'])
    var_len_features = set(['diagnoses', 'procedures', 'labs', 'prescriptions'])
    mimic_embeddings = {'gender': 1, 'has_dod': 1, 'has_icu_stay': 1,
                        'admission_type': 3, 'diagnoses': 40, 'procedures': 40,
                        'labs': 40, 'prescriptions': 40}
