import torch
from models import VGG19, Decoder
from utils import eval_transform, extract_image_names, load_image, deprocess
from ops import style_decorator
import argparse
import os

parser = argparse.ArgumentParser(description='Avatar Net')

parser.add_argument('--content-path', type=str, help='path to the content image')
parser.add_argument('--style-path', type=str, help='path to the style image')
parser.add_argument('--content-dir', type=str, help='path to the content image folder')
parser.add_argument('--style-dir', type=str, help='path to the style image folder')

parser.add_argument('--decoder-path', type=str, default='./model/model.pth')

parser.add_argument('--kernel-size', type=int, default=7)
parser.add_argument('--stride', type=int, default=1)
parser.add_argument('--alpha', type=float, default=0.8)
parser.add_argument('--synthesis', type=int, default=0, help='0-transfer, 1-synthesis')

parser.add_argument('--save-dir', type=str, default='./results')
parser.add_argument('--save-name', type=str, default='result', help='save name for single output image')
parser.add_argument('--save-ext', type=str, default='jpg', help='The extension name of the output image')

parser.add_argument('--content-size', type=int, default=768, help='New (minimum) size for the content image')
parser.add_argument('--style-size', type=int, default=768, help='New (minimum) size for the style image')
parser.add_argument('--gpu', type=int, default=0, help='ID of the GPU to use; for CPU mode set --gpu = -1')

args = parser.parse_args()

assert args.content_path is not None or args.content_dir is not None, \
    'Either --content-path or --content-dir should be given.'
assert args.style_path is not None or args.style_dir is not None, \
    'Either --style-path or --style-dir should be given.'

device = torch.device('cuda:%s' % args.gpu if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

encoder = VGG19().to(device)
decoder = Decoder(pretrained_path=args.decoder_path).to(device)


def style_transfer(content, style):

    c_features = encoder(content)
    s_features = encoder(style)

    z_cs = style_decorator(c_features.relu4_1, s_features.relu4_1, args.kernel_size, args.stride, args.alpha)
    output = decoder(z_cs, s_features)

    return output


if not os.path.exists(args.save_dir):
    print('Creating save folder at', args.save_dir)
    os.mkdir(args.save_dir)

content_paths = []
style_paths = []

if args.content_dir:
    # use a batch of content images
    content_paths = extract_image_names(args.content_dir)
else:
    # use a single content image
    content_paths.append(args.content_path)

if args.style_dir:
    # use a batch of style images
    style_paths = extract_image_names(args.style_dir)
else:
    # use a single style image
    style_paths.append(args.style_path)

print('Number content images:', len(content_paths))
print('Number style images:', len(style_paths))

c_transform = eval_transform(args.content_size)
s_transform = eval_transform(args.style_size)

with torch.no_grad():

    for i in range(len(content_paths)):
        content = load_image(content_paths[i])
        content = c_transform(content).unsqueeze(0)
        content = content.to(device)

        for j in range(len(style_paths)):
            style = load_image(style_paths[j])
            style = s_transform(style).unsqueeze(0)
            style = style.to(device)

            if args.synthesis == 0:
                output = style_transfer(content, style)
                output = deprocess(output)

                if len(content_paths) == 1 and len(style_paths) == 1:
                    # used a single content and style image
                    save_path = '%s/%s.%s' % (args.save_dir, args.save_name, args.save_ext)
                else:
                    # used a batch of content and style images
                    save_path = '%s/%s_%s.%s' % (args.save_dir, i, j, args.save_ext)

                print('Output image saved at:', save_path)
                output.save(save_path)
            else:
                content = torch.rand(*content.shape).uniform_(0, 1).to(device)
                for iteration in range(3):
                    output = style_transfer(content, style)
                    content = output
                    output = deprocess(output)

                    if len(content_paths) == 1 and len(style_paths) == 1:
                        # used a single content and style image
                        save_path = '%s/%s_%s.%s' % (args.save_dir, args.save_name, iteration, args.save_ext)
                    else:
                        # used a batch of content and style images
                        save_path = '%s/%s_%s_%s.%s' % (args.save_dir, i, j, iteration, args.save_ext)

                    print('Output image saved at:', save_path)
                    output.save(save_path)
