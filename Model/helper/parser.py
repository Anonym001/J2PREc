import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run JPPREC.")

    parser.add_argument('--data_path', nargs='?', default='../Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='brightkite',
                        help='Choose a dataset')
    parser.add_argument('--weights_path', nargs='?', default='',
                        help='Store model path.')
    parser.add_argument('--embed_size', type=int, default=64,
                        help='Embedding size.')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size.')
    parser.add_argument('--epoch', type=int, default=400,
                        help='Number of epoch.')
    parser.add_argument('--layer_size', nargs='?', default='[64]',
                        help='Output sizes for every propagation layer')
    parser.add_argument('--regs', nargs='?', default='[1e-5,1e-5,1e-5]',
                        help='L2 Regularization')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate.')
    parser.add_argument('--proj_path', nargs='?', default='',
                        help='Project path.')
    parser.add_argument('--verbose', type=int, default=5,
                        help='Interval of evaluation.')
    parser.add_argument('--item_size', type=int, default=64,
                        help='Embedding size.')
    parser.add_argument('--batch_size_item', type=int, default=1024,
                        help='batch size.')
    parser.add_argument('--adj_type', nargs='?', default='si',
                        help='{bi, si}.')
    parser.add_argument('--alg_type', nargs='?', default='jpprec',
                        help='Specify model name')
    parser.add_argument('--adj_uni_type', nargs='?', default='sum',
                        help='Specify a loss type (uni, sum).')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='Choose a gpu')
    parser.add_argument('--node_dropout', nargs='?', default='[0.1,0.1,0.1]',
                        help='dropout probability')
    parser.add_argument('--mess_dropout', nargs='?', default='[0.1,0.1,0.1]',
                        help='dropout probability')
    parser.add_argument('--Ks', nargs='?', default='[3,5,10,15,20]',
                        help='Output sizes of every layer')
    parser.add_argument('--save_flag', type=int, default=1,
                        help='1: save model parameters or 0 otherwise')
    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='mini-batch testing')

    return parser.parse_args()
