import argparse


def get_ptb_args():
    """ ArgumentParser for ptb dataset. """
    parser = argparse.ArgumentParser(description='training the network on ptb.')

    # Multi-source
    parser.add_argument('--multi_source', type=bool, default=False)
    parser.add_argument('--source_domain', type=str, default='EWT GUM LinES ParTUT')
    parser.add_argument('--target_domain', type=str, default='EWT')
    parser.add_argument('--data_dir', type=str, default='data/ud-v2.2-eng')

    # Data
    parser.add_argument('--emb',
                        default='data/Embed/glove/glove.6B/glove.6B.100d.txt',
                        help='pretrained_embeddings_file')
    parser.add_argument('--use_bert', type=bool, default=False)
    parser.add_argument('--dropout_emb', default=0.33, type=float)
    parser.add_argument('--train_data_path', default='./data/PTB/train.conllx')
    parser.add_argument('--dev_data_path', default='./data/PTB/dev.conllx')
    parser.add_argument('--test_data_path', default='./data/PTB/test.conllx')

    # Save
    parser.add_argument('--best_model_dir', default='best_model/PTB/Dozat')
    parser.add_argument('--test_result_dir', default='./result/PTB/Dozat')
    parser.add_argument('--log_dir', default='./log/PTB/CNN')

    # Network
    parser.add_argument('--word_dims', default=100, type=int)
    parser.add_argument('--tag_dims', default=100, type=int)
    parser.add_argument('--rel_dims', default=100, type=int)
    parser.add_argument('--lstm_layers', default=3, type=int)
    parser.add_argument('--lstm_hiddens', default=400, type=int)
    parser.add_argument('--mlp_arc_size', default=500, type=int)
    parser.add_argument('--mlp_rel_size', default=100, type=int)
    parser.add_argument('--dropout_mlp', default=0.33, type=float)
    parser.add_argument('--dropout_lstm_input', default=0.33, type=float)
    parser.add_argument('--dropout_lstm_hidden', default=0.33, type=float)

    # Attention
    parser.add_argument('--use_atten', default=False, type=bool)
    parser.add_argument('--num_heads_atten', default=4, type=int)

    # Transoformer
    parser.add_argument('--num_heads_transformer', default=8, type=int)
    parser.add_argument('--transformer_layers', default=4, type=int)
    parser.add_argument('--use_transformer', default=False, type=bool)
    parser.add_argument('--dim_feedforward', default=800, type=int)

    # CNN
    parser.add_argument('--use_cnn', default=False, type=bool)
    parser.add_argument('--cnn_hiddens', default=800, type=int)

    # Optimizer
    parser.add_argument('--lr', type=float, default=0.002, help='learning_rate')
    parser.add_argument('--beta_1', type=float, default=0.9)
    parser.add_argument('--beta_2', type=float, default=0.9)
    parser.add_argument('--epsilon', type=float, default=1e-12)
    parser.add_argument('--decay', type=float, default=0.75)
    parser.add_argument('--decay_epoch', type=int, default=3)
    parser.add_argument('--earlystop_epoch', type=int, default=10)

    # Run
    parser.add_argument('--train_batch_size', default=64, type=int)
    parser.add_argument('--test_batch_size', default=32, type=int)
    parser.add_argument('--gpu', default=2, type=int)
    parser.add_argument('--num_epochs', default=500, type=int)

    args = parser.parse_args()
    return args


def get_args():
    """ ArgumentParser for single domain """
    parser = argparse.ArgumentParser(description='training your network')

    # Data
    parser.add_argument('--emb',
                        default='data/Embed/glove/glove.6B/glove.6B.100d.txt',
                        help='pretrained_embeddings_file')
    parser.add_argument('--dropout_emb', default=0.33, type=float)
    parser.add_argument('--data_path', default='./data/ud-v2.2-eng')
    parser.add_argument('--source_domain', default='EWT GUM LinES')
    parser.add_argument('--target_domain', default='ParTUT')

    # Save
    parser.add_argument('--best_model_dir', default='./best_model/in-domain')
    parser.add_argument('--test_result_dir', default='./test_result/in-domain')
    parser.add_argument('--log_dir', default='./log/in-domain')

    # Network
    parser.add_argument('--word_dims', default=100, type=int)
    parser.add_argument('--tag_dims', default=100, type=int)
    parser.add_argument('--rel_dims', default=100, type=int)
    parser.add_argument('--lstm_layers', default=3, type=int)
    parser.add_argument('--lstm_hiddens', default=400, type=int)
    parser.add_argument('--mlp_arc_size', default=500, type=int)
    parser.add_argument('--mlp_rel_size', default=100, type=int)
    parser.add_argument('--dropout_mlp', default=0.33, type=float)
    parser.add_argument('--dropout_lstm_input', default=0.33, type=float)
    parser.add_argument('--dropout_lstm_hidden', default=0.33, type=float)

    # Optimizer
    parser.add_argument('--lr',
                        type=float,
                        default=0.002,
                        help='learning_rate')

    # Run
    parser.add_argument('--train_batch_size', default=128, type=int)
    parser.add_argument('--test_batch_size', default=128, type=int)
    parser.add_argument('--gpu', default=1, type=int)
    parser.add_argument('--num_epochs', default=300, type=int)

    args = parser.parse_args()
    return args


def get_meta_args():
    parser = argparse.ArgumentParser(
        description='ArgumentParser for meta-learning model')

    # Data
    parser.add_argument('--emb',
                        default='data/Embed/glove/glove.6B/glove.6B.100d.txt',
                        help='pretrained_embeddings_file')
    parser.add_argument('--dropout_emb', default=0.33, type=float)
    parser.add_argument('--data_path', default='./data/ud-v2.2-eng')
    parser.add_argument('--source_domain', default="EWT GUM LinES")
    parser.add_argument('--target_domain', default='ParTUT')

    # Save
    parser.add_argument('--best_model_dir', default='./meta_best_model')
    parser.add_argument('--test_result_dir', default='./meta_test_result')
    parser.add_argument('--log_dir', default='./meta_log')

    # Network
    parser.add_argument('--word_dims', default=100, type=int)
    parser.add_argument('--tag_dims', default=100, type=int)
    parser.add_argument('--lstm_layers', default=2, type=int)
    parser.add_argument('--lstm_hiddens', default=300, type=int)
    parser.add_argument('--mlp_arc_size', default=200, type=int)
    parser.add_argument('--mlp_rel_size', default=100, type=int)
    parser.add_argument('--dropout_mlp', default=0.33, type=float)
    parser.add_argument('--dropout_lstm_input', default=0.33, type=float)
    parser.add_argument('--dropout_lstm_hidden', default=0.33, type=float)

    # Optimizer
    parser.add_argument('--lr_learner',
                        type=float,
                        default=0.01,
                        help='learning_rate')
    parser.add_argument('--lr_meta',
                        type=float,
                        default=0.001,
                        help='learning_rate')
    parser.add_argument('--lr_decay', type=float, default=0.95)

    # Run
    parser.add_argument('--train_batch_size', default=64, type=int)
    parser.add_argument('--test_batch_size', default=64, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--num_epochs', default=5000, type=int)

    # Meta
    parser.add_argument('--N', default=2, type=int, help='n_way')
    parser.add_argument('--K', default=64, type=int, help='k_shot')
    parser.add_argument('--num_updates', default=1, type=int)
    parser.add_argument('--tune_num_epochs', default=20, type=int)

    args = parser.parse_args()
    return args
