from source import Source
from model_initializer import ModelInitializer
from visualizer import Visualizer
import cv2


def main():
    sourceList=[
        {'processing_mode':'opencv', 'source_type':'webcam', 'source_path':0},
        {'processing_mode':'opencv', 'source_type':'youtube', 'source_path':'https://www.youtube.com/shorts/F-Yqrdy6oHU'},
        {'processing_mode':'opencv', 'source_type':'videofile', 'source_path':'/home/dhananjay/Desktop/SARWESH-SIR-TASK/WhatsApp Video 2024-10-29 at 3.33.59 PM.mp4'},
        # {'processing_mode':'opencv', 'source_type':'webcam', 'source_path':0},
        ]
    
    modelList=[
        {'model_type':'yolov8_detection', 'model_path':'yolov8n.pt'},
        {'model_type':'yolov8_detection', 'model_path':'yolov8n.pt'},
        {'model_type':'yolov8_detection','model_path':'yolov8n.pt'},
        # {'model_type':'yolov8_detection', 'model_path':'yolov8n.pt'},
    ]

    visualizationList=[
        {'task':'detection'},
        {'task':'detection'},
        {'task':'detection'},
        # {'task':'detection'},
    ]

    # Print alignment check
    # for idx in range(len(sourceList)):
    #     print(f"Source {idx}: {sourceList[idx]['source_type']}, Model: {modelList[idx]['model_type']}, Task: {visualizationList[idx]['task']}")



    sourceObjectList=[Source(processing_mode=src['processing_mode'], source_type=src['source_type'], source_path=src['source_path']) for src in sourceList]
    modelObjectList=[ModelInitializer(model_type=idx['model_type'],model_path=idx['model_path']).get_model() for idx in modelList]
    visualizer=Visualizer()

    while True:
        # Dict to store frame and results by idx for each src
        frames_dict={}

        for idx,src in enumerate(sourceObjectList):
            if src.cap.isOpened():
                frame=src.get_frame()
                if frame is not None:
                    model=modelObjectList[idx] #Getting specific model from modelObjectList as per idx
                    result=model.predict(frame)
                    frames_dict[idx]=(frame,result)
                else:
                    frames_dict[idx]=(None,None)

        # Check if all frames are None, meaning all sources are exhausted
        if all(frame is None for frame, _ in frames_dict.values()):
            break

        # Display results for each frame in separate windows based on their index and task
        for idx, (frame, results) in frames_dict.items():
            if frame is not None and results is not None:
                # Use the specific task for visualization from visualizationList
                task = visualizationList[idx]['task']
                print(f"{idx}:{task}")
                visualizer.display_results(
                    frame,
                    results,
                    task=task,
                    names=getattr(modelObjectList[idx].model, 'names', None)
                )
                # Assign a unique window name for each source using its index
                cv2.imshow(f'Source {idx}', frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release all sources and close windows
    for source in sourceObjectList:
        source.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

    