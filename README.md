# Multi-Source Model Processing Pipeline

This pipeline is designed to handle multiple video sources and models for **object detection**, **segmentation**, and **classification** tasks. It supports a variety of video input sources such as **webcam**, **YouTube URLs**, **video files**, and other **URLs**. The pipeline operates with separate threads for each source for efficient parallel processing, while the **visualizer** runs on the main thread for all sources.

## 🚀 Features

- **Multiple Source Support**: Input sources can be **webcam**, **video files**, **URLs**, or **YouTube URLs**.
- **Model Flexibility**: Currently supports **YOLO detection**, **segmentation**, and **classification models**.
- **Multithreading**: Each video source runs in its own thread, enabling parallel processing.
- **Centralized Visualization**: Visualization tasks are handled in the main thread for all sources.
- **Easy Configuration**: Minimal changes are needed to configure the pipeline for new models or sources.

## 📋 Requirements

Before running the pipeline, ensure you have the following dependencies installed:

- opencv-python
- ultralytics
- pytube
- yt_dlp


To install the required dependencies, run the following command:

```bash
pip install -r requirements.txt
```


## Project structure

```bash
├── cameraReader.py           # Handles video frame capturing from sources
├── model.py                  # Contains different model classes (YOLO detection, segmentation, classification)
├── model_initializer.py      # Initializes the selected model based on user input
├── visualizer.py             # Handles visualization and result display
├── main.py                   # Runs the entire pipeline, supporting multiple sources
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

## File-content guide
1. source.py: contains code for the differeny source option.
2. model.py: contains code for the models. For adding model in our project we should add model and predict code of the new model in this file.
3. model_initialization.py: initializes model based on chosen option.
4. visualizer.py: contains code for visualization as per selected task.
5. main.py: contains code which canm support only single source at a time.
6. scratch.py: conatins code which supports multiple sources at a time but doesn't contain threading so error in reciving frames from one source can fail the model.
7. threadTest.py: conatins code which suports multiple sources and threading is added. So here error in one source will not affect other ones.

