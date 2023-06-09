# +
import os
import time
import numpy as np
import cv2
from rknn.api import RKNN
import argparse


# -

def parse_args():
    parser = argparse.ArgumentParser(description='Run convert onnx to rknn')
    parser.add_argument('onnx', help='path to onnx model')
    parser.add_argument('--rknn', help='create path to rknn model', default=None)
    parser.add_argument('--verbose', help='show conversion progress',
                       type=bool, default=False)
    parser.add_argument('--q', help='switch quantization', 
                                type=bool, default=False)
    parser.add_argument('--dataset', default=None,
                        help='path to .txt file with list of preprocessed images')
    parser.add_argument('--mean', help='mean value of the color palette',
                        type=list, default=[123.675, 116.28, 103.53])
    parser.add_argument('--std', help='color palette standard deviation',
                        type=list, default=[58.395, 57.12, 57.375])
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # Get parameters
    args = parse_args()
    ONNX_MODEL = args.onnx
    RKNN_MODEL = args.rknn
    QUANTIZE_ON = args.q
    DATASET = args.dataset
    VERBOSE = args.verbose
    MEAN = args.mean
    STD = args.std
    if not RKNN_MODEL:
        filename, file_extension = os.path.splitext(ONNX_MODEL)
        if QUANTIZE_ON:
            RKNN_MODEL = filename + '_quant.rknn'
        else:
            RKNN_MODEL = filename + '.rknn'
        
    # Create RKNN object
    rknn = RKNN(verbose=VERBOSE)

    # pre-process config
    print('--> Config model')
    rknn.config(mean_values=[MEAN], 
                std_values=[STD], 
                target_platform='rk3588')
    print('done')

    # Load ONNX model
    print('--> Loading model')
    ret = rknn.load_onnx(model=ONNX_MODEL)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=QUANTIZE_ON, dataset=DATASET)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export RKNN model
    print('--> Export rknn model')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    rknn.release()


