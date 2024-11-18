# Weapon Detection System

This repository contains a Python implementation of a weapon detection system using the YOLO framework. The model identifies weapons in video streams or images and applies bounding boxes around detected objects.

## Features
- Real-time object detection using YOLO.
- Configurable for custom YOLO weights and configuration files.
- Adaptive frame rate processing for optimized performance.
- Gaussian blurring applied to detected weapons for moderation purposes.

## Requirements
Install the dependencies using `pip`:
```bash
pip install opencv-python numpy
