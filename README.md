# labelImgWithOpenPose
LabelImgWithOpenPose is a graphical human pose image annotation tool development based on labelImg and OpenPose.

## Reference:

### labelIng: ```https://github.com/tzutalin/labelImg```

### OpenPose: ```https://github.com/CMU-Perceptual-Computing-Lab/openpose```

## Requierment:
### OpenPose: https://github.com/CMU-Perceptual-Computing-Lab/openpose

## Quick Start:
1. Clone source
```
git clone https://github.com/zeuspnt/labelImg.git
```

2. Install lib for labelImg
Python3 and Qt5
```
sudo apt-get install pyqt5-dev-tools
sudo pip3 install lxml
make qt5py3
```
See more install tutorial from labelImg ```https://github.com/tzutalin/labelImg#installation```

3. Install OpenPose

https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation.md

Run ```make install``` after build successfully.

4. Edit some information

Open file ```labelImgOpenPose.py``` and edit at line 78
```
# Ensure you point to the correct path where models are located
params["default_model_folder"] = "folder/path/to/models/in/openpose"
```
usually, it will at ```.../openpose/models/```

5. Run
```
python3 labelImgOpenPose.py
```
## Extract keypoints from OpenPose to txt file (extractKeypointsWithOpenPose.py)

### Edit some parameter

Set GPU ID to use if you have multi GPU
```
params["num_gpu_start"] = 0
```

Set the path to models folder of OpenPose
```
params["default_model_folder"] = "/path/to/openpose/models/"
```

### Run with argument

```
python extractKeypointsWithOpenPose.py --imagedir=/home/ai/cuda-workspace/labelImg/examples/images/
```

Get all images in path at parameter ```--imagedir``` to pass into Openpose and save keypoints in txt file in keypoints directory with structure:
```
examples
|
---- images
|  |
|   ---- COCO_val2014_000000000474.jpg
---- keypoints
   |
    ---- COCO_val2014_000000000474.txt
```

## Review result be extracted (labelImgOpenPoseTXT.py)

```
python labelImgOpenPoseTXT.py
```

Click on ```Open Dir``` button to open folder contain images, the app will automate load txt files in keypoints folder and draw that pose on images.
