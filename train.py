import torch
from torch import nn
import torch.utils.data as data
from models import VGG19, Decoder
from utils import train_transform, eval_transform, Dataset, InfiniteSampler, deprocess, extract_image_names, load_image
from ops import TVloss, learning_rate_decay
import argparse
from tqdm import tqdm
import os


parser = argparse.ArgumentParser(description='Avatar Net')

# Basic options
parser.add_argument('--dataset-dir', type=str, required=True, help='Directory containing images for training')
parser.add_argument('--model-save-dir', type=str, default='./models')
parser.add_argument('--test-dataset-dir', type=str, default='./test_data')
parser.add_argument('--test-save-dir', type=str, default='./test_outputs')
parser.add_argument('--gpu', type=int, default=0, help='ID of the GPU to use; for CPU mode set --gpu = -1')
parser.add_argument('--nThreads', type=int, default=12)

# Preprocessing options
parser.add_argument('--final-size', type=int, default=256, help='Size of images used for training')
parser.add_argument('--image-size', type=int, default=512, help='Size of images before cropping')

# Training options
parser.add_argument('--batch-size', type=int, default=2)
parser.add_argument('--learning-rate', type=float, default=0.001)
parser.add_argument('--learning-rate-decay', type=float, default=0)
parser.add_argument('--max-iter', type=int, default=160000)
parser.add_argument('--tv-weight', type=float, default=1.)
parser.add_argument('--feature-weight', type=float, default=0.1)

# Verbosity
parser.add_argument('--eval-loss-every', type=int, default=500)
parser.add_argument('--print-every', type=int, default=500)
parser.add_argument('--save-every', type=int, default=1000)
parser.add_argument('--test-every', type=int, default=1000)

args = parser.parse_args()

device = torch.device('cuda:%s' % args.gpu if torch.cuda.is_available() and args.gpu != 1 else 'cpu')

encoder = VGG19().to(device)
decoder = Decoder().to(device)

train_transform = train_transform(args.image_size, args.final_size)
test_transform = eval_transform(args.image_size)

dataset = Dataset(args.dataset_dir, train_transform)

data_loader = iter(data.DataLoader(
    dataset,
    batch_size=args.batch_size,
    num_workers=args.nThreads,
    sampler=InfiniteSampler(dataset)
))

optimizer = torch.optim.Adam(decoder.parameters(), lr=args.learning_rate)

criterion = nn.MSELoss()

loss_buff = 0
momentum = 2 / (1 + args.eval_loss_every)

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

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    loss_buff = momentum * total_loss.item() + (1 - momentum) * loss_buff
    lr = learning_rate_decay(optimizer, args.learning_rate, global_step, args.learning_rate_decay)

    if global_step % args.print_every == 0:
        tqdm.write('step: %d, loss: %f, lr: %f' % (global_step, loss_buff, lr))

    if global_step % args.save_every == 0:
        if not os.path.exists(args.model_save_dir):
            os.mkdir(args.model_save_dir)
        save_path = '%s/%s_%f.pth' % (args.model_save_dir, global_step, loss_buff)
        torch.save(decoder.state_dict(), save_path)

    if global_step % args.test_every == 0:
        if not os.path.exists(args.test_save_dir):
            os.mkdir(args.test_save_dir)

        with torch.no_grad():

            image_paths = extract_image_names(args.test_dataset_dir)

            for i in range(len(image_paths)):
                test_input = load_image(image_paths[i])
                test_input = test_transform(test_input)
                test_input = test_input.unsqueeze(0).to(device)

                inputs_features = encoder(test_input)
                output = decoder(inputs_features.relu4_1, inputs_features)

                output = deprocess(output)
                save_path = '%s/%s_%s.jpg' % (args.test_save_dir, global_step, i)
                output.save(save_path)
