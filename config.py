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
    bidirectional = False # TODO
    pretrained_emb = True
    conditional = True
    attention = False
    fixed_len_features = ['word', 'gender', 'has_dod', 'has_icu_stay', 'admission_type']
    var_len_features = ['diagnoses', 'procedures', 'labs', 'prescriptions']
    mimic_embeddings = {'gender': 1, 'has_dod': 1, 'has_icu_stay': 1,
                        'admission_type': 3,
                        'diagnoses': 15,
                        'procedures': 15,
                        'labs': 15,
                        'prescriptions': 15}
    max_aux_values = {'diagnoses': 40,
                      'procedures': 25,
                      'labs': 200,
                      'prescriptions': 150} # XXX remove if not needed

    def __init__(self):
        if self.conditional:
            self.data_size = len(self.fixed_len_features)
        else:
            self.data_size = 1
