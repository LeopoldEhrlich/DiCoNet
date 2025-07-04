import matplotlib.pyplot as plt
import os
import argparse

from main import test
from DCN import DivideAndConquerNetwork
from data_generator import Generator

hull_dims = 2

parser = argparse.ArgumentParser()
parser.add_argument('--num_examples_test', nargs='?', const=1, type=int,
                    default=256)
parser.add_argument('--batch_size', nargs='?', const=1, type=int, default=64)
parser.add_argument('--path_dataset', nargs='?', const=1, type=str, default='./data')
parser.add_argument('--path', nargs='?', const=1, type=str, default='./reg')

parser.add_argument('--dynamic', action='store_true', default=True,
                    help='Use DCN. If not set, run baseline. (depth=0)')


###############################################################################
#                             PtrNet arguments                                #
###############################################################################

parser.add_argument('--load_merge', nargs='?', const=1, type=str, default='./reg')
parser.add_argument('--num_units_merge', nargs='?', const=1, type=int,
                    default=512)
parser.add_argument('--rnn_layers', nargs='?', const=1, type=int, default=1)
parser.add_argument('--grad_clip_merge', nargs='?', const=1,
                    type=float, default=2.0)
parser.add_argument('--merge_sample', action='store_true',
                    help='Sample inputs to connect different PtrNets when'
                    'cascading PtrNets')

###############################################################################
#                              split arguments                                #
###############################################################################

parser.add_argument('--load_split', nargs='?', const=1, type=str,default='./reg')
parser.add_argument('--split_layers', nargs='?', const=1, type=int, default=5)
parser.add_argument('--num_units_split', nargs='?', const=1, type=int,
                    default=15)
parser.add_argument('--grad_clip_split', nargs='?', const=1,
                    type=float, default=40.0)
parser.add_argument('--regularize_split', action='store_true',
                    help='regularize the split training with variance prior.')
parser.add_argument('--beta', nargs='?', const=1, type=float, default=1.0)
parser.add_argument('--random_split', action='store_true',
                    help='Do not train split. Take uniform random samples')

args = parser.parse_args()

def test_scale_invariancy():
    DCN = DivideAndConquerNetwork(hull_dims, args.batch_size,
                                  args.num_units_merge, args.rnn_layers,
                                  args.grad_clip_merge,
                                  args.num_units_split, args.split_layers,
                                  args.grad_clip_split, beta=args.beta)
    
    DCN.load_split(args.load_split)
    DCN.load_merge(args.load_merge)

    DCN.cuda()

    gen = Generator(0, args.num_examples_test,
                    args.path_dataset, args.batch_size,scales_test=list(range(1,8)))
    
    gen.load_dataset()

    DCN.batch_size = args.batch_size
    DCN.merge.batch_size = args.batch_size
    DCN.split.batch_size = args.batch_size

    accuracies_test, miss_rates = test(DCN, gen, args)

    plt.figure()
    plt.plot(range(len(accuracies_test)), accuracies_test, 'ro-')
    plt.title('Test Accuracies')    
    plt.xlabel('Max Depth')
    plt.ylabel('Accuracy')      
    plt.savefig(os.path.join(args.path, 'test_accuracies.png'))

    plt.figure()
    plt.plot(range(len(miss_rates)), miss_rates, 'ro-')
    plt.title('Test Miss rates')    
    plt.xlabel('Max Depth')
    plt.ylabel('Miss Rate')      
    plt.savefig(os.path.join(args.path, 'test_miss_rates.png'))

    plt.figure(figsize=(10, 6))

    # Plot both metrics on the same axes
    line1, = plt.plot(range(len(accuracies_test)), accuracies_test, 'b-o', label='Accuracy', linewidth=2, markersize=8)
    line2, = plt.plot(range(len(miss_rates)), miss_rates, 'r--s', label='Miss Rate', linewidth=2, markersize=8)

    plt.title('Model Performance vs. Max Depth', fontsize=14, pad=20)    
    plt.xlabel('Max Depth', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.xticks(range(len(accuracies_test)), fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add legend
    plt.legend(handles=[line1, line2], fontsize=12)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save the figure
    plt.savefig(os.path.join(args.path, 'performance_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()

    plt.close()    

def test_base_case():
    DCN = DivideAndConquerNetwork(hull_dims, args.batch_size,
                                  args.num_units_merge, args.rnn_layers,
                                  args.grad_clip_merge,
                                  args.num_units_split, args.split_layers,
                                  args.grad_clip_split, beta=args.beta)
    
    DCN.load_split(args.load_split)
    DCN.load_merge(args.load_merge)

    DCN.cuda()

    gen = Generator(0, 10_000,
                    args.path_dataset, args.batch_size,scales_test=[2])
    
    gen.compute_length = lambda scale, mode: (2 if scale == 1 else 3, 2 if scale == 1 else 3)
    gen.load_dataset('base')

    DCN.batch_size = args.batch_size
    DCN.merge.batch_size = args.batch_size
    DCN.split.batch_size = args.batch_size

    accuracies_test, miss_rates = test(DCN, gen, args)

if __name__ == "__main__":
    test_scale_invariancy()
    #test_base_case()