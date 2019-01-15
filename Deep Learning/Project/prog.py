import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--square", help="display a square of a given number",
                    type=int)
parser.add_argument("--arch", help="choose your architecture", type=str, default="vgg16")
parser.add_argument("--learning_rate", help="choose learning rate", type=str)
parser.add_argument("--gpu", action='store_true', default=True)
args = parser.parse_args()
# print(args.square**2)
print(args.arch)
print(args.learning_rate)
print(args.gpu)
