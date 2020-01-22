import torch
from torch import nn
import torch.utils.data as data
from models import VGG19, Decoder
from utils import train_transform, Dataset, InfiniteSampler, deprocess
from ops import TVloss
import argparse
from tqdm import tqdm


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
parser.add_argument('--max-iter', type=int, default=160000)
parser.add_argument('--tv-weight', type=float, default=1.)
parser.add_argument('--feature-weight', type=float, default=0.1)

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

criterion = nn.MSELoss()

for global_step in tqdm(range(args.max_iter)):

    inputs = next(data_loader).to(device)

    inputs_features = encoder(inputs)
    reconstructed_inputs = decoder(inputs_features.relu4_1, inputs_features)

    pixel_loss = criterion(inputs, reconstructed_inputs)

    reconstructed_features = encoder(reconstructed_inputs)

    feature_loss = 0
    for inp_f, rec_f in zip(inputs_features, reconstructed_features):
        feature_loss += 1/4 * criterion(inp_f, rec_f)

    tv_loss = TVloss(reconstructed_inputs, args.tv_weight)

    total_loss = pixel_loss + args.feature_weight * feature_loss + tv_loss
