import argparse
import splitfolders

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-d", "--dataset_directory",
                    type=str, 
                    required=True,
                    help="Dataset Directory")

parser.add_argument("-o","--output_directory",
                    type = str,
                    default='./Train_Val_Dataset',
                    help="Output Directory for Dataset")

args = parser.parse_args()

if __name__ == '__main__':
    splitfolders.ratio(args.dataset_directory, output=args.output_directory,
                        seed=1337, ratio=(.8, .2), group_prefix=None, move=False)
