from source import Source
from model_initializer import ModelInitializer
from visualizer import Visualizer
import cv2
import threading
import queue
import time

# Define a function that captures frames and processes them in a separate thread
def process_source(idx, src, model, task, result_queue, stop_event):
    while not stop_event.is_set():  # Check for stop signal
        if not src.cap.isOpened():
            break  # Break if the source is closed
        
        frame = src.get_frame()
        if frame is None:
            break  # Break if there are no more frames

        # Run model prediction
        result = model.predict(frame)

        # Put frame and result in the queue for display
        result_queue.put((idx, frame, result, task))
        
        # To avoid high CPU usage, add a small delay
        time.sleep(0.01)

    # Release the source when done
    src.release()
    result_queue.put((idx, None, None, task))  # Signal end of processing for this source

def main():
    sourceList = [
        {'processing_mode': 'opencv', 'source_type': 'webcam', 'source_path': 0},
        {'processing_mode': 'opencv', 'source_type': 'youtube', 'source_path': 'https://www.youtube.com/shorts/F-Yqrdy6oHU'},
        {'processing_mode': 'opencv', 'source_type': 'videofile', 'source_path': '/home/dhananjay/Desktop/SARWESH-SIR-TASK/WhatsApp Video 2024-10-29 at 3.33.59 PM.mp4'},
    ]
    
    modelList = [
        {'model_type': 'yolov8_detection', 'model_path': 'yolov8n.pt'},
        {'model_type': 'yolov8_detection', 'model_path': 'yolov8n.pt'},
        {'model_type': 'yolov8_detection', 'model_path': 'yolov8n.pt'},
    ]

    visualizationList = [
        {'task': 'detection'},
        {'task': 'detection'},
        {'task': 'detection'},
    ]

    # Initialize source, model, and visualizer objects
    sourceObjectList = [Source(processing_mode=src['processing_mode'], source_type=src['source_type'], source_path=src['source_path']) for src in sourceList]
    modelObjectList = [ModelInitializer(model_type=idx['model_type'], model_path=idx['model_path']).get_model() for idx in modelList]
    visualizer = Visualizer()

    # Queue to store results for display in the main thread
    result_queue = queue.Queue()

    # Create a stop event to signal threads to exit
    stop_event = threading.Event()

    # Create and start a thread for each source
    threads = []
    for idx, (src, model, vis_task) in enumerate(zip(sourceObjectList, modelObjectList, visualizationList)):
        task = vis_task['task']
        thread = threading.Thread(target=process_source, args=(idx, src, model, task, result_queue, stop_event))
        thread.start()
        threads.append(thread)

    # Main loop for displaying frames
    while True:
        # Check if there are results to display
        while not result_queue.empty():
            idx, frame, result, task = result_queue.get()

            if frame is None:
                # If frame is None, it signals that this source is done processing
                continue

            # Display the frame with results
            visualizer.display_results(frame, result, task=task, names=getattr(modelObjectList[idx].model, 'names', None))
            cv2.imshow(f'Source {idx}', frame)

        # Check for 'q' key to exit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            stop_event.set()  # Signal all threads to stop
            break

    # Wait for all threads to finish
    for thread in threads:
        thread.join()

    # Release any OpenCV resources and close windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
