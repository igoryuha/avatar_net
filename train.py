import torch
from models import VGG19, Decoder
import argparse


parser = argparse.ArgumentParser(description='Avatar Net')

parser.add_argument('--content-dir', type=str, required=True, help='Directory containing content images for training')
parser.add_argument('--gpu', type=int, default=0, help='ID of the GPU to use; for CPU mode set --gpu = -1')
parser.add_argument('--nThreads', type=int, default=12)

args = parser.parse_args()

device = torch.device('cuda:%s' % args.gpu if torch.cuda.is_available() and args.gpu != 1 else 'cpu')


encoder = VGG19().to(device)
decoder = Decoder().to(device)
