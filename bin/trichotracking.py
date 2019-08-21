import argparse
import sys
from os.path import join
import os
sys.path.remove(os.path.dirname(__file__))

from trichotracking.processing import Processor


def parse_args(arguments):
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', help='source folder of images', required=True)
    parser.add_argument('--px', help='px length in [µm/px]', required=True, type=int)
    parser.add_argument('--dest', help='Result directory')
    parser.add_argument('--plot', help='Flag indicating if picutures are plotted', type=bool, default=False)
    parser.add_argument('--dark', help='Flag indicating if images are darkfield', type=bool, default=False)
    parser.add_argument('--blur', help='Flag indicating if images should be blurred', type=bool, default=True)
    parser.add_argument('--dLink', help='Maximal linking distance in px', type=int, default=10)
    parser.add_argument('--dMerge', help='Maximal merging distance in px', type=int, default=10)
    parser.add_argument('--dMergeBox', help='Maximal merging distance of minimal boxes in px', type=int, default=10)
    parser.add_argument('--kChamber', help='Kernel size to erode chamber', type=int, default=400)
    parser.add_argument('--dt', help='Image sequence capture time', type=int)
    args = parser.parse_args(arguments[1:])

    # dargs = vars(args)
    # for key in dargs.keys():
    #     print("{}: {}, type = {}".format(key, dargs[key], type(dargs[key])))

    srcDir = args.src
    px = args.px
    dest = join(args.src, 'results') if args.dest is None else args.dest
    plot = args.plot
    dark = args.dark
    blur = args.blur
    dLink = args.dLink
    dMerge = args.dMerge
    dMergeBox = args.dMergeBox
    kChamber = args.kChamber
    dt = args.dt

    return srcDir, px, dest, plot, dark, blur, dLink, dMerge, dMergeBox, kChamber, dt


if __name__ == '__main__':
    arguments = sys.argv
    processor = Processor(*parse_args(arguments))
