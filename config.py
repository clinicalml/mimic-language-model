import tensorflow as tf

flags = tf.flags

# command-line config
flags.DEFINE_string ("data_path",          "/home/ankit/devel/data/MIMIC3pk", "Data path")
flags.DEFINE_string ("save_file",          "savemodel.dat", "Save file")
flags.DEFINE_string ("load_file",          "loadmodel.dat", "File to load model from")
flags.DEFINE_float  ("init_scale",         0.05, "Variable initialization scale")
flags.DEFINE_float  ("learning_rate",      1e-3, "ADAM learning rate")
flags.DEFINE_float  ("max_grad_norm",      5,    "Gradient clipping")
flags.DEFINE_integer("num_layers",         2,    "Number of LSTM layers")
flags.DEFINE_integer("num_steps",          15,   "Number of steps to unroll")
flags.DEFINE_integer("hidden_size",        650,  "LSTM state size")
flags.DEFINE_integer("learn_wordemb_size", 150,  "Number of learnable dimensions in word embeddings")
flags.DEFINE_integer("max_epoch",          6,    "Maximum number of epochs to run for")
flags.DEFINE_float  ("keep_prob",          0.5,  "Dropout keep probability")
flags.DEFINE_integer("batch_size",         20,   "Batch size")
flags.DEFINE_integer("print_every",        200,  "Print every these many steps")
flags.DEFINE_integer("save_every",         500,  "Save every these many steps") # TODO change
flags.DEFINE_bool   ("pretrained_emb",     True, "Use pretrained embeddings")
flags.DEFINE_bool   ("conditional",        True, "Use a conditional language model")


class Config(object):
    # additional config
    fixed_len_features = set(['gender', 'has_dod', 'has_icu_stay', 'admission_type'])
    var_len_features = set(['diagnoses', 'procedures', 'labs', 'prescriptions'])
    mimic_embeddings = {'gender': 1, 'has_dod': 1, 'has_icu_stay': 1,
                        'admission_type': 3, 'diagnoses': 40, 'procedures': 40,
                        'labs': 40, 'prescriptions': 40}

    def __init__(self):
        for k, v in flags.FLAGS.__dict__['__flags'].items():
            setattr(self, k, v)
