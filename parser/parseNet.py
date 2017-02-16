#!/usr/bin/env python
"""
Draw a graph of the net architecture.
"""
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from google.protobuf import text_format

import caffe
import caffe.draw
from caffe.proto import caffe_pb2


def parse_args():
    """Parse input arguments
    """

    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('input_net_proto_file',
                        help='Input network prototxt file')
    parser.add_argument('output_text_file',
                        help=('Output text file'),
			default="out.txt")
    parser.add_argument('--phase',
                        help=('Which network phase to draw: can be TRAIN, '
                              'TEST, or ALL.  If ALL, then all layers are drawn '
                              'regardless of phase.'),
                        default="ALL")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    net = caffe_pb2.NetParameter()
    text_format.Merge(open(args.input_net_proto_file).read(), net)
    print('Printing net to %s' % args.output_text_file)
    phase=None;
    if args.phase == "TRAIN":
        phase = caffe.TRAIN
    elif args.phase == "TEST":
        phase = caffe.TEST
    elif args.phase != "ALL":
        raise ValueError("Unknown phase: " + args.phase)
#caffe.draw.draw_net_to_file(net, args.output_image_file, args.rankdir,phase)
    print(text_format.MessageToString(net))

if __name__ == '__main__':
    main()
