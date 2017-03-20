#!/usr/bin/env python
"""
Draw a graph of the net architecture.
"""
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from google.protobuf import text_format

from caffe.proto import caffe_pb2


def ListLayers(net_prototxt):
   for layer in net_prototxt.layer:
      if layer.type == "Data": 
         print "data layers name : ", layer.name
      elif layer.type == "Convolution":
         print "conv layer name: ",layer.name
      elif layer.type == "Pooling":
	 print "Pooling layers name : ", layer.name
      elif layer.type == "InnerProduct":
	 print "IP layer name : ",layer.name


def parse_args():
    """Parse input arguments
    """

    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_net_proto_file',
                        help='Input network prototxt file')
    parser.add_argument('output_text_file',
                        help='Output text file')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    net = caffe_pb2.NetParameter()
    fin = open(args.input_net_proto_file,"rb")
#net.ParseFromString(fin.read())
#fin.close()
#ListLayers(net)
    
    text_format.Merge(fin.read(), net)
    fin.close()
    ListLayers(net)
#print('Printing net to %s' % args.output_text_file)
#output = open(args.output_text_file,"w")
#output.write(text_format.MessageToString(net))
#output.close()
    
if __name__ == '__main__':
    main()
