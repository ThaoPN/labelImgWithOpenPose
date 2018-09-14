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

from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement
from lxml import etree
import codecs

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
params["disable_blending"] = True
# Ensure you point to the correct path where models are located
params["default_model_folder"] = "/home/ai/cuda-workspace/openpose/models/"
# Construct OpenPose object allocates GPU memory
openpose = op.OpenPose(params)

IMAGE_EXTENTIONS = ['.bmp', '.cur', '.gif', '.icns', '.ico', '.jpeg', '.jpg', '.pbm', '.pgm', '.png', '.ppm', '.svg', '.svgz', '.tga', '.tif', '.tiff', '.wbmp', '.webp', '.xbm', '.xpm']
XML_EXT = '.xml'
TXT_EXT = '.txt'
ENCODE_METHOD = 'utf-8'

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

def prettify(elem):
    """
        Return a pretty-printed XML string for the Element.
    """
    rough_string = ElementTree.tostring(elem, 'utf8')
    root = etree.fromstring(rough_string)
    return etree.tostring(root, pretty_print=True, encoding=ENCODE_METHOD).replace("  ".encode(), "\t".encode())
    # minidom does not support UTF-8
    '''reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="\t", encoding=ENCODE_METHOD)'''

def saveKeypointsToXML(targetFile, keypoints):
    if not targetFile.endswith('xml'):
        return None
    
    top = Element('annotation')
    
    bndboxes = []
    
    for key in keypoints:
        XY = []
        X = []
        Y = []
        
        for p in key:
            x, y = int(p[0]), int(p[1])
            XY.append(x)
            XY.append(y)
            if x > 0:
                X.append(x)
            if y > 0:
                Y.append(y)
        
        xmin = min(X)
        xmax = max(X)
        ymin = min(Y)
        ymax = max(Y)
        
        each_object = {'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax':ymax, 'keypoints': ', '.join(map(str, XY))}
        bndboxes.append(each_object)
    
    for each_object in bndboxes:
        object_item = SubElement(top, 'object')
        
        keypoints_item = SubElement(object_item, 'keypoints')
        keypoints_item.text = each_object['keypoints']
        
        bndbox = SubElement(object_item, 'bndbox')
        xmin = SubElement(bndbox, 'xmin')
        xmin.text = str(each_object['xmin'])
        ymin = SubElement(bndbox, 'ymin')
        ymin.text = str(each_object['ymin'])
        xmax = SubElement(bndbox, 'xmax')
        xmax.text = str(each_object['xmax'])
        ymax = SubElement(bndbox, 'ymax')
        ymax.text = str(each_object['ymax'])
    
    out_file = None
    if targetFile is None:
        out_file = codecs.open(
            self.filename + XML_EXT, 'w', encoding=ENCODE_METHOD)
    else:
        out_file = codecs.open(targetFile, 'w', encoding=ENCODE_METHOD)

    prettifyResult = prettify(top)
    out_file.write(prettifyResult.decode('utf8'))
    out_file.close()    

def saveKeypointsOfImage(imagePath, keypoints, outputType):
    dirPathContaint = os.path.split(os.path.dirname(imagePath))[0]
    
    nameWithoutExt = os.path.basename(os.path.splitext(imagePath)[0])
    
    dirPathSavedKeypoints = os.path.join(dirPathContaint, "keypoints")
    if not os.path.isdir(dirPathSavedKeypoints):
        os.mkdir(dirPathSavedKeypoints)
        
    if outputType == 'xml':
        keypointsFileName = os.path.join(dirPathSavedKeypoints, nameWithoutExt) + XML_EXT
        saveKeypointsToXML(keypointsFileName, keypoints)
    else:
        keypointsFileName = os.path.join(dirPathSavedKeypoints, nameWithoutExt) + TXT_EXT

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
    parser.add_argument('--imagedir', type=str, default='/home/ai/cuda-workspace/openpose/examples/media')
    parser.add_argument('--outputtype', type=str, default='txt')
    args = parser.parse_args()
    
    outputType = args.outputtype
    
    imagedir = args.imagedir
    if imagedir != '' :
        images = scanAllImages(args.imagedir)
        for image in images:
            keypoints = getKeypointsOfImage(image)
            saveKeypointsOfImage(image, keypoints, outputType)
