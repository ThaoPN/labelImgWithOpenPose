import sys
import os
import numpy as np
import cv2
import common
from common import CocoPart
import string
import random
import argparse
from libs.ustr import ustr

sys.path.append('/usr/local/python')
try:
    from openpose import openpose as op
except:
    raise Exception('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')

params = dict()
params["logging_level"] = 3
params["output_resolution"] = "-1x-1"
params["net_resolution"] = "-1x368"
params["model_pose"] = "COCO"
params["alpha_pose"] = 0.6
params["scale_gap"] = 0.3
params["scale_number"] = 1
params["render_threshold"] = 0.05
# If GPU version is built, and multiple GPUs are available, set the ID here
params["num_gpu_start"] = 0
params["disable_blending"] = False
# Ensure you point to the correct path where models are located
params["default_model_folder"] = "/home/ai/cuda-workspace/openpose/models/"
# Construct OpenPose object allocates GPU memory
openpose = op.OpenPose(params)

IMAGE_EXTENTIONS = ['.bmp', '.cur', '.gif', '.icns', '.ico', '.jpeg', '.jpg', '.pbm', '.pgm', '.png', '.ppm', '.svg', '.svgz', '.tga', '.tif', '.tiff', '.wbmp', '.webp', '.xbm', '.xpm']

def scanAllImages(folderPath):
    images = []

    for root, dirs, files in os.walk(folderPath):
        for file in files:
            if file.lower().endswith(tuple(IMAGE_EXTENTIONS)):
                relativePath = os.path.join(root, file)
                path = ustr(os.path.abspath(relativePath))
                images.append(path)
        # Just get images in 1 level of this directory
        break
    images.sort(key=lambda x: x.lower())
    return images
def getKeypointsOfImage(imagePath):
    img = cv2.imread(imagePath)
    keypoints = openpose.forward(img, False)[:,:,:-1]
    # Remove accuracy colunm and return
    return keypoints

def saveKeypointsOfImage(imagePath, keypoints):
    dirPathContaint = os.path.split(os.path.dirname(imagePath))[0]
    
    nameWithoutExt = os.path.basename(os.path.splitext(imagePath)[0])
    
    dirPathSavedKeypoints = os.path.join(dirPathContaint, "keypoints")
    if not os.path.isdir(dirPathSavedKeypoints):
        os.mkdir(dirPathSavedKeypoints)
        
    keypointsFileName = os.path.join(dirPathSavedKeypoints, nameWithoutExt) + '.txt'

    allKeypointsString = ""
    for key in keypoints:
        keypointsString = ""
        for pointTutle in key:
            keypointsString += "{}, {}, ".format(pointTutle[0], pointTutle[1])
        allKeypointsString += keypointsString[:-2] + "\n"
    
    with open(keypointsFileName, 'w') as f:
        f.write(allKeypointsString)           
            
    print(keypointsFileName)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract keypoints from an image and save it to a text file with same name')
    parser.add_argument('--image', type=str, default='')
    parser.add_argument('--imagedir', type=str, default='/home/ai/cuda-workspace/openpose/examples/media')
    args = parser.parse_args()
    
    imagedir = args.imagedir
    if imagedir != '' :
        images = scanAllImages(args.imagedir)
        for image in images:
            keypoints = getKeypointsOfImage(image)
            saveKeypointsOfImage(image, keypoints)
            