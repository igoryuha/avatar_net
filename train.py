import torch
import torch.utils.data as data
from models import VGG19, Decoder
from utils import train_transform, Dataset, InfiniteSampler, deprocess
import argparse


parser = argparse.ArgumentParser(description='Avatar Net')

# Basic options
parser.add_argument('--dataset-dir', type=str, required=True, help='Directory containing images for training')
parser.add_argument('--gpu', type=int, default=0, help='ID of the GPU to use; for CPU mode set --gpu = -1')
parser.add_argument('--nThreads', type=int, default=12)

# Preprocessing options
parser.add_argument('--final-size', type=int, default=256, help='Size of images used for training')
parser.add_argument('--image-size', type=int, default=512, help='Size of images before cropping')

# Training options
parser.add_argument('--batch-size', type=int, default=2)

args = parser.parse_args()

device = torch.device('cuda:%s' % args.gpu if torch.cuda.is_available() and args.gpu != 1 else 'cpu')

encoder = VGG19().to(device)
decoder = Decoder().to(device)

train_transform = train_transform(args.image_size, args.final_size)
dataset = Dataset(args.dataset_dir, train_transform)

data_loader = iter(data.DataLoader(
    dataset,
    batch_size=args.batch_size,
    num_workers=args.nThreads,
    sampler=InfiniteSampler(dataset)
))

