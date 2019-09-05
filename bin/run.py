import argparse
import sys
from os.path import abspath, join

from trichotracking.processing import Processor


def parse_args(arguments):
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', help='source folder of images', required=True)
    parser.add_argument('--px', help='px length in [µm/px]', required=True, type=int)
    parser.add_argument('--expId', help='Unique experiment identifier', required=True)
    parser.add_argument('--dest', help='Result directory')
    parser.add_argument('--plot', help='Flag indicating if picutures are plotted', type=bool, default=False)
    parser.add_argument('--dark', help='Flag indicating if images are darkfield', type=bool, default=False)
    parser.add_argument('--blur', help='Flag indicating if images should be blurred', type=bool, default=True)
    parser.add_argument('--thresh', help='Segmentation threshold', type=int, default=30)
    parser.add_argument('--dLink', help='Maximal linking distance in px', type=int, default=10)
    parser.add_argument('--dMerge', help='Maximal merging distance in px', type=int, default=10)
    parser.add_argument('--dMergeBox', help='Maximal merging distance of minimal boxes in px', type=int, default=10)
    parser.add_argument('--kChamber', help='Kernel size to erode chamber', type=int, default=400)
    parser.add_argument('--dt', help='Image sequence capture time', type=float)
    args = parser.parse_args(arguments[1:])

    srcDir = abspath(args.src)
    px = args.px
    expId = args.expId
    dest = join(srcDir, 'results') if args.dest is None else args.dest
    plot = args.plot
    dark = args.dark
    blur = args.blur
    thresh = args.thresh
    dLink = args.dLink
    dMerge = args.dMerge
    dMergeBox = args.dMergeBox
    kChamber = args.kChamber
    dt = args.dt

    return srcDir, px, expId, dest, plot, dark, blur, thresh, dLink, dMerge, dMergeBox, kChamber, dt


if __name__ == '__main__':
    arguments = sys.argv
    processor = Processor(*parse_args(arguments))
