from source import Source
from model_initializer import ModelInitializer
from visualizer import Visualizer
import cv2

def main():
    # Define multiple sources
    sources = [
        {'pm':'opencv','type': 'youtube', 'path': 'https://www.youtube.com/shorts/F-Yqrdy6oHU'},
        {'pm':'ffmpeg','type': 'webcam', 'path': 0}
    ]
    model_type = 'yolov8_detection'
    model_path = 'yolov8n.pt'

    # Initialize model
    model_initializer = ModelInitializer(model_type=model_type, model_path=model_path)
    model = model_initializer.get_model()

    # Initialize visualizer
    visualizer = Visualizer()

    # Initialize sources and store their handlers
    source_objects = [Source(processing_mode=src['pm'], source_type=src['type'], source_path=src['path']) for src in sources]

    # Run loop for each source
    while True:
        frames = []
        results_list = []
        
        # Capture frame for each source
        for idx, source in enumerate(source_objects):
            if source.cap.isOpened():
                frame = source.get_frame()
                if frame is not None:
                    frames.append(frame)
                    results_list.append(model.predict(frame))
                else:
                    frames.append(None)
                    results_list.append(None)

        # Check if all frames are None, meaning all sources are exhausted
        if all(f is None for f in frames):
            break

        # Display results for each frame in separate windows
        for idx, (frame, results) in enumerate(zip(frames, results_list)):
            if frame is not None and results is not None:
                visualizer.display_results(
                    frame, 
                    results, 
                    task=model_type.split('_')[1], 
                    names=getattr(model.model, 'names', None)
                )
                # Assign a unique window name for each source
                cv2.imshow(f'Source {idx}', frame)
        
        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release all sources and close windows
    for source in source_objects:
        source.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
