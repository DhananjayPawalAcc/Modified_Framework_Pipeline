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

