# Object detection with [Owl-ViT](https://github.com/google-research/scenic/tree/main/scenic/projects/owl_vit)

This project implements the Owl-ViT model for zero-shot object detection in videos (or several images).

data/video.mp4

## Contents

Table of contents:
- [Installation](#installation)
- [Usage](#usage)
- [Example](#example)

## Installation

Run the following commands:

```
git clone https://github.com/killian31/ObjectsDetection.git
cd ObjectsDetection
pip install -r requirements.txt
```

## Usage

```
Usage: 
    python3 owl_vit.py [-h] --imgs_dir IMGS_DIR --save_to SAVE_TO [--process_video] [--video_filename VIDEO_FILENAME] [--image_start IMAGE_START] [--image_end IMAGE_END]
                       [--texts TEXTS [TEXTS ...]] [--thresholds THRESHOLDS [THRESHOLDS ...]] [--box_thickness BOX_THICKNESS] [--save_model]

Options:
  -h, --help            show this help message and exit
  --imgs_dir IMGS_DIR   The directory containing the images.
  --save_to SAVE_TO     the directory in which to save the processed images
  --fps FPS             Number of frames per second for the output video (default: None, determined automatically if --process_video).
  --process_video       Wether to get images from a video or not (default: False).
  --video_filename VIDEO_FILENAME
                        Name of video file to process (default: None).
  --image_start IMAGE_START
                        Frame to start from (default:0).
  --image_end IMAGE_END
                        Frame to end with (default: last (0).
  --texts TEXTS [TEXTS ...]
                        A list of texts to detect in the images (default: 14 random texts).
  --colors COLORS [COLORS ...]
                        A list of (b,g,r) colors for the bounding boxes (default: 14 different colors).
  --thresholds THRESHOLDS [THRESHOLDS ...]
                        A list of thresholds between 0 and 1 (default: 14 low thresholds for the texts).
  --box_thickness BOX_THICKNESS
                        The thickness of the bounding boxes to draw (default: 2).
  --save_model          Whether to save the pretrained model locally or not (default: False).

```

## Example

```
python3 owl_vit.py --imgs_dir path/to/store/images --save_to output_name --process_video --video_filename MyVideo.avi --texts person ball shoe --thresholds 0.07 0.15 0.03 --save_model
```
