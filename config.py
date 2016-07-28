import tensorflow as tf

flags = tf.flags

# command-line config
flags.DEFINE_string ("data_path",          "data",  "Data path")
flags.DEFINE_string ("save_file",          "models/recent.dat", "Save file")
flags.DEFINE_string ("load_file",          "",      "File to load model from")
flags.DEFINE_float  ("learning_rate",      1e-3,    "ADAM learning rate")
flags.DEFINE_float  ("max_grad_norm",      5,       "Gradient clipping")
flags.DEFINE_integer("num_layers",         2,       "Number of LSTM layers")
flags.DEFINE_integer("num_steps",          20,      "Number of steps to unroll for RNNs")
flags.DEFINE_integer("context_size",       5,       "Context size for feedforward nets")
flags.DEFINE_integer("hidden_size",        500,     "Hidden state size")
flags.DEFINE_integer("learn_wordemb_size", 150,     "Number of learnable dimensions in word " \
                                                    "embeddings")
flags.DEFINE_integer("max_steps",          9999999, "Maximum number of steps to run for")
flags.DEFINE_integer("max_epoch",          6,       "Maximum number of epochs to run for")
flags.DEFINE_integer("softmax_samples",    1000,    "Number of classes to sample for softmax")
flags.DEFINE_float  ("keep_prob",          1.0,     "Dropout keep probability")
flags.DEFINE_float  ("struct_keep_prob",   0.5,     "Structural info dropout keep probability")
flags.DEFINE_integer("batch_size",         25,      "Batch size")
flags.DEFINE_integer("print_every",        500,     "Print every these many steps")
flags.DEFINE_integer("inspect_every",      -1,      "Inspect a batch every these many batches " \
                                                    "during testing, -1 to disable")
flags.DEFINE_integer("save_every",         10000,   "Save every these many steps")
flags.DEFINE_bool   ("pretrained_emb",     True,    "Use pretrained embeddings")
flags.DEFINE_bool   ("conditional",        True,    "Use a conditional language model")
flags.DEFINE_bool   ("training",           True,    "Training mode, turn off for testing")
flags.DEFINE_bool   ("recurrent",          False,   "Use a recurrent language model")

flags.DEFINE_integer("dims_gender",         1,   "Dimensionality for gender")
flags.DEFINE_integer("dims_has_dod",        1,   "Dimensionality for has_dod")
flags.DEFINE_integer("dims_has_icu_stay",   1,   "Dimensionality for has_icu_stay")
flags.DEFINE_integer("dims_admission_type", 4,   "Dimensionality for admission_type")
flags.DEFINE_integer("dims_diagnoses",      100, "Dimensionality for diagnoses")
flags.DEFINE_integer("dims_procedures",     100, "Dimensionality for procedures")
flags.DEFINE_integer("dims_labs",           100, "Dimensionality for labs")
flags.DEFINE_integer("dims_prescriptions",  100, "Dimensionality for prescriptions")


class Config(object):
    mimic_embeddings = {}

    # additional config
    fixed_len_features = set(['gender', 'has_dod', 'has_icu_stay', 'admission_type'])
    var_len_features = set(['diagnoses', 'procedures', 'labs', 'prescriptions'])
    testing_splits = range(1)
    training_splits = range(1,100)


    def __init__(self):
        for k, v in flags.FLAGS.__dict__['__flags'].items():
            setattr(self, k, v)
            if k.startswith('dims_'):
                self.mimic_embeddings[k[len('dims_'):]] = v

        if not self.recurrent:
            self.num_steps = self.context_size # reuse the num_steps config for FF
