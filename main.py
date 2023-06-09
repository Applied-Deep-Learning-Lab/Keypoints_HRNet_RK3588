import os
from rknnlite.api import RKNNLite
import argparse
import cv2
import time
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Run a pose model')
    parser.add_argument('model', help='weights file path')
    parser.add_argument('image', help='image file path')
    parser.add_argument('--size', help='the size of input image', 
                                type=int, default=512)
    parser.add_argument('--num-clusses', help='the number of classes',
                                type=int, default=17)
    parser.add_argument('--work-dir', help='the dir to save result', 
                                default='./results')
    parser.add_argument('--time', help='show inference & postprocess time', 
                                default=False)
    args = parser.parse_args()
    return args

def transform_preds(coords, width, model_size):
    """Get final keypoint predictions from heatmaps and apply scaling and
    translation to map them back to the image.

    Args:
        coords (np.ndarray[k, ndims]):

        width (int): size of output predictions
        model_size (int): input size of model

    Returns:
        np.ndarray: Predicted coordinates in the images.
    """
    
    scale_x = model_size / width
    scale_y = model_size / width
    target_coords = np.ones_like(coords)
    target_coords[:, :, 0] = coords[:, :, 0] * scale_x
    target_coords[:, :, 1] = coords[:, :, 1] * scale_y
    
    return target_coords

def _get_max_preds(heatmaps):
    """Get keypoint predictions from score maps.
    Note:
        batch size: N
        num_keypoints: K
        heatmap height: H
        heatmap width: W

    Args:
        heatmaps (np.ndarray[N, K, H, W]): model predicted heatmaps.

    Returns:
        tuple: A tuple containing aggregated results.
        - preds (np.ndarray[N, K, 2]): Predicted keypoint location.
        - maxvals (np.ndarray[N, K, 1]): Scores (confidence) of the keypoint
    """
    assert isinstance(heatmaps, np.ndarray), (
        'heatmaps should be numpy.ndarray')
    assert heatmaps.ndim == 4, 'batch images should be 4-ndim'
    
    N, K, H, W = heatmaps.shape
    heatmaps_reshaped = heatmaps.reshape((N, K, -1)) # [H][W]: [1,1,3],[2,2,4] -> [1,1,3,2,2,4]
    idx = np.argmax(heatmaps_reshaped, 2).reshape((N, K, 1)) # indexes of first max number
    maxvals = np.amax(heatmaps_reshaped, 2).reshape((N, K, 1)) # values of first max number

    preds = np.tile(idx, (1,1,2)).astype(np.float32)
    preds[:,:,0] = preds[:,:,0] % W # x of max_score pixel
    preds[:,:,1] = preds[:,:,1] // W # y of max_score pixel
    
    preds = np.where(np.tile(maxvals, (1,1,2)) > 0.0, preds, -1)
    return preds, maxvals

def keypoints_from_heatmaps(heatmaps, model_size):
    # Avoid being affected
    heatmaps = heatmaps.copy()

    N, K, H, W = heatmaps.shape
    preds, maxvals = _get_max_preds(heatmaps)
    preds = transform_preds(preds, W, model_size)
    return preds, maxvals

def decode(outputs, model_size, num_clusses, batch_size=1):
    preds, maxvals = keypoints_from_heatmaps(outputs, model_size)
    all_preds = np.zeros((batch_size, num_clusses, 3), dtype=np.float32)
    all_preds[:, :, 0:2] = preds[:, :num_clusses, 0:2]
    all_preds[:, :, 2:3] = maxvals[:,:num_clusses,:]
    return all_preds

def draw(bgr, predict_dict, out_dir, show_labels=False):
    for all_pred in predict_dict:
        for x,y,s in all_pred:
            cv2.circle(bgr, (int(x), int(y)), 3, (0, 255, 120), -1)
    num = len(os.listdir(out_dir))
    cv2.imwrite(f"{out_dir}/result_{num}.jpg", bgr)


if __name__ == "__main__":
    # Get parameters
    args = parse_args()
    RKNN_MODEL = args.model
    IMG_PATH = args.image
    OUT_DIR = args.work_dir
    MODEL_SIZE = args.size
    NUM_CLUSSES = args.num_clusses
    TIME = args.time

    # Create RKNN object
    rknn = RKNNLite()

    # Load RKNN model
    print("--> Loading model")
    if not os.path.exists(RKNN_MODEL):
        print("model not exist")
        exit(-1)
    ret = rknn.load_rknn(RKNN_MODEL)
    if ret != 0:
        print("Load rknn model failed!")
        exit(ret)
    print("done")

    # init runtime environment
    print("--> Init runtime environment")
    ret = rknn.init_runtime()
    if ret != 0:
        print("Init runtime environment failed")
        exit(ret)
    print('done')

    # Load image
    src_img = cv2.imread(IMG_PATH)
    img = cv2.resize(src_img, [MODEL_SIZE,MODEL_SIZE])
    height, width, _ = img.shape

    # Inference
    print("--> Running model")
    start = time.time()
    outputs = rknn.inference(inputs=[img])[0]
    end = time.time()
    runTime = end - start
    runTime_ms = runTime * 1000
    if TIME:
        print("Inference Time:", runTime_ms, "ms")
    
    if not os.path.isdir(OUT_DIR):
        os.mkdir(OUT_DIR)

    # Post Process
    print("--> Running postprocess")
    start = time.time()
    predict_dict = decode(outputs, MODEL_SIZE, NUM_CLUSSES)
    out_img = draw(img, predict_dict, OUT_DIR)
    end = time.time()
    runTime = end - start
    runTime_ms = runTime * 1000
    if TIME:
        print("Postprocess Time:", runTime_ms, "ms")
    print("- Complete -")


