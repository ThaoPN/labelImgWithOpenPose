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

# Path to graph of pose classifier
POSE_CLASSIFIER_GRAPH_PATH = 'pose_classifier_graph/pose_classifier.h5.pb'

LABELS = [    
    "standing",
    "bending",
    "crouching"
]

def loadGraph():
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(POSE_CLASSIFIER_GRAPH_PATH, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it 
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph

def normKeypoint(X):
    num_sample = X.shape[0]
    # Keypoints
    Nose = X[:,0*2:0*2+2]
    Neck = X[:,1*2:1*2+2]
    RShoulder = X[:,2*2:2*2+2]
    RElbow = X[:,3*2:3*2+2]
    RWrist = X[:,4*2:4*2+2]
    LShoulder = X[:,5*2:5*2+2]
    LElbow = X[:,6*2:6*2+2]
    LWrist = X[:,7*2:7*2+2]
    RHip = X[:,8*2:8*2+2]
    RKnee = X[:,9*2:9*2+2]
    RAnkle = X[:,10*2:10*2+2]
    LHip = X[:,11*2:11*2+2]
    LKnee = X[:,12*2:12*2+2]
    LAnkle = X[:,13*2:13*2+2]
    REye = X[:,14*2:14*2+2]
    LEye = X[:,15*2:15*2+2]
    REar = X[:,16*2:16*2+2]
    LEar = X[:,17*2:17*2+2]

    # Length of head
    length_head      = np.sqrt(np.square((LEye[:,0:1]+REye[:,0:1])/2 - Neck[:,0:1]) + np.square((LEye[:,1:2]+REye[:,1:2])/2 - Neck[:,1:2]))

    # Length of torso
    length_torso     = np.sqrt(np.square(Neck[:,0:1]-(LHip[:,0:1]+RHip[:,0:1])/2) + np.square(Neck[:,1:2]-(LHip[:,1:2]+RHip[:,1:2])/2))

    # Length of right leg
    length_leg_right = np.sqrt(np.square(RHip[:,0:1]-RKnee[:,0:1]) + np.square(RHip[:,1:2]-RKnee[:,1:2])) \
    + np.sqrt(np.square(RKnee[:,0:1]-RAnkle[:,0:1]) + np.square(RKnee[:,1:2]-RAnkle[:,1:2]))

    # Length of left leg
    length_leg_left = np.sqrt(np.square(LHip[:,0:1]-LKnee[:,0:1]) + np.square(LHip[:,1:2]-LKnee[:,1:2])) \
    + np.sqrt(np.square(LKnee[:,0:1]-LAnkle[:,0:1]) + np.square(LKnee[:,1:2]-LAnkle[:,1:2]))

    # Length of leg
    length_leg = np.maximum(length_leg_right, length_leg_left)

    # Length of body
    length_body = length_head + length_torso + length_leg
    length_body[length_body == 0] = 1
    # The center of gravity
    centr_x = X[:, 0::2].sum(1).reshape(num_sample,1) / 18LABELS = [    
    "standing",
    "bending",
]
    centr_y = X[:, 1::2].sum(1).reshape(num_sample,1) / 18

    # The  coordinates  are  normalized relative to the length of the body and the center of gravity
    X_norm_x = (X[:, 0::2] - centr_x) / length_body
    X_norm_x[X_norm_x <= -1] = 0
    X_norm_x[X_norm_x >= 1] = 0

    X_norm_y = (X[:, 1::2] - centr_y) / length_body
    X_norm_y[X_norm_y <= -1] = 0
    X_norm_y[X_norm_y >= 1] = 0
    
    X_norm = np.column_stack((X_norm_x[:,:1], X_norm_y[:,:1]))
    
    for i in range(1, X.shape[1]//2):
        X_norm = np.column_stack((X_norm, X_norm_x[:,i:i+1], X_norm_y[:,i:i+1]))
    
    return X_norm


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

def saveKeypointsToXML(sess, targetFile, keypoints):
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
        
        norm_XY = np.array([XY], dtype=np.float32)
        poseIndexs = sess.run(tf.argmax(y,1), feed_dict={x: norm_XY})
        pose = LABELS[poseIndexs[0]]
        
        each_object = {'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax':ymax, 'keypoints': ', '.join(map(str, XY)), 'pose': pose}
        bndboxes.append(each_object)
    
    for each_object in bndboxes:
        object_item = SubElement(top, 'object')
        
        keypoints_item = SubElement(object_item, 'keypoints')
        keypoints_item.text = each_object['keypoints']
        
        pose_item = SubElement(object_item, 'pose')
        pose_item.text = each_object['pose']
        
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

def saveKeypointsOfImage(sess, imagePath, keypoints):
    dirPathContaint = os.path.split(os.path.dirname(imagePath))[0]
    
    nameWithoutExt = os.path.basename(os.path.splitext(imagePath)[0])
    
    dirPathSavedKeypoints = os.path.join(dirPathContaint, "keypoints")
    if not os.path.isdir(dirPathSavedKeypoints):
        os.mkdir(dirPathSavedKeypoints)
        
    keypointsFileName = os.path.join(dirPathSavedKeypoints, nameWithoutExt) + XML_EXT
    saveKeypointsToXML(keypointsFileName, keypoints)
            
    print(keypointsFileName)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract keypoints from an image and save it to a text file with same name')
    parser.add_argument('--imagedir', type=str, default='examples/images')
    args = parser.parse_args()
    
    outputType = args.outputtype
    
    imagedir = args.imagedir
    if imagedir != '' :
        images = scanAllImages(args.imagedir)
        with tf.Session(graph=graph) as sess:
            for image in images:
                keypoints = getKeypointsOfImage(image)
                saveKeypointsOfImage(sess, image, keypoints)
                    
