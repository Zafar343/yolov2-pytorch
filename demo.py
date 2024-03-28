import os
import shutil
import argparse
import time
import torch
from torch.autograd import Variable
from PIL import Image
from test import prepare_im_data
# from yolov2 import Yolov2
from yolov2_tiny_2 import Yolov2
from yolo_eval import yolo_eval
# from util.visualize import draw_detection_boxes
# import matplotlib.pyplot as plt
from util.network import WeightLoader
import cv2
import numpy as np


def parse_args():

    parser = argparse.ArgumentParser('Yolo v2')
    parser.add_argument('--output_dir', dest='output_dir',
                        default='output', type=str)         # specify the path of output label folder here in default
    parser.add_argument('--model_name', dest='model_name',
                        default='data/pretrained/yolov2-tiny-voc.pth', type=str) # specift the path of weights here in default
    parser.add_argument('--cuda', dest='use_cuda',
                        default=False, type=bool)
    parser.add_argument('--data', type=str,
                        default=None,
                        help='Path to data.txt file containing images list') # specify the data file (name should be data.txt) which should be a text file containing list of paths of all the inference images

    parser.add_argument('--classes', type=str,
                        default=None,
                        help='provide the list of class names if other than voc') # Example 'default = ('Vehicle', 'Rider', 'Pedestrian')'
    parser.add_argument('--conf_thres', type=float,
                        default=0.001,
                        help='choose a confidence threshold for inference, must be in range 0-1')
    parser.add_argument('--nms_thres', type=float,
                        default=0.45,
                        help='choose an nms threshold for post processing predictions, must be in range 0-1s')
    args = parser.parse_args()
    return args

def demo():
    args = parse_args()
    print('call with args: {}'.format(args))

    if args.data==None:

        # input images
        images_dir   = 'images'
        images_names = ['image1.jpg', 'image2.jpg']
    else:
        with open(args.data, 'r') as f:
            images_names = f.readlines()
    
    try: classes
    except: classes=None
    
    if classes is None:
        classes = ('aeroplane', 'bicycle', 'bird', 'boat',
                                'bottle', 'bus', 'car', 'cat', 'chair',
                                'cow', 'diningtable', 'dog', 'horse',
                                'motorbike', 'person', 'pottedplant',
                                'sheep', 'sofa', 'train', 'tvmonitor')
    else:
        classes = args.classes        
    
    # set the save folder path for predictions
    if args.data==None:
        pred_dir = os.path.join(images_dir, 'preds')    
    else:
        dir      = args.data.split('/data')[0]
        pred_dir = os.path.join(dir, 'preds')                    
    
    if not os.path.exists(pred_dir):
        print(f'making {pred_dir}')
        os.mkdir(pred_dir)
    else:
        print('Deleting existing pred dir')
        shutil.rmtree(pred_dir, ignore_errors=True)
        print(f'making new {pred_dir}')
        os.mkdir(pred_dir)
    
    model = Yolov2(classes=classes)
    
    model_type = args.model_name.split('.')[-1]
    
    if model_type == 'weights':
        weight_loader = WeightLoader()
        weight_loader.load(model, 'yolo-voc.weights')
        print('loaded')

    else:
        model_path = args.model_name
        print('loading model from {}'.format(model_path))
        if torch.cuda.is_available():
            checkpoint = torch.load(model_path, map_location=('cuda:0'))
        else:
            checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])

    if args.use_cuda:
        model.cuda()

    model.eval()
    print('model loaded')

    for image_name in images_names:
        if args.data==None:
            image_path = os.path.join(images_dir, image_name)
            img        = Image.open(image_path)
        else:
            image_path = image_name.split('\n')[0]
            img        = Image.open(image_path)   
        
        im_data, im_info = prepare_im_data(img)

        if args.use_cuda:
            im_data_variable = Variable(im_data).cuda()
        else:
            im_data_variable = Variable(im_data)

        tic = time.time()

        yolo_output = model(im_data_variable)
        yolo_output = [item[0].data for item in yolo_output]
        detections  = yolo_eval(yolo_output, im_info, conf_threshold=args.conf_thres, nms_threshold=args.nms_thres)

        toc = time.time()
        cost_time = toc - tic
        print('im detect, cost time {:4f}, FPS: {}'.format(
            toc-tic, int(1 / cost_time)))

        
        if len(detections)>0:
            name        = image_name.split('/')[-1]
            pred_name   = name.split('.')[0] + '.txt'
            pred_path   = os.path.join(pred_dir, pred_name)
            det_boxes   = detections[:, :5].cpu().numpy()
            det_classes = detections[:, -1].long().cpu().numpy()
            pred        = np.zeros((det_boxes.shape[0],6))
            pred[:, :5] = det_boxes
            pred[:,-1]  = det_classes # pred is [x, y, w, h, conf, cls]
            
            preds = []
            
            for i in range(pred.shape[0]):
                _pred      = pred[i]
                label_line = str(int(_pred[-1])) + ' {} {} {} {} {} \n'.format(round(_pred[0], 7),
                                                        round(_pred[1], 8),
                                                        round(_pred[2], 8),
                                                        round(_pred[3], 8),
                                                        round(_pred[4], 8)) # line formate is (category, x, y, width, height)
                preds.append(label_line)
            # write the predictions of current image to the write ;ocation
            f = open(pred_path, 'w')
            f.writelines(preds)            
            print()

if __name__ == '__main__':
    demo()
