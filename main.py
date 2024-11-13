from source import Source
from model_initializer import ModelInitializer
from visualizer import Visualizer
import cv2

def main():
    # User selects the input source and model type
    source_type = 'videofile'
    source_path='/home/dhananjay/Desktop/SARWESH-SIR-TASK/WhatsApp Video 2024-10-29 at 3.33.59 PM.mp4'
    # source_path  = 'rtsp://192.168.1.111'    #LICESNSE DETECTION CAM URL AND PLS CONNECT TO THE SAME NETWORK TO WHICH THIS CAM IS CONNECTED 
    model_type = 'yolov8_segmentation'  # Choose from 'yolov8_detection', 'yolov8_segmentation', 'yolov8_classification'
    model_path='yolov8n-seg.pt'
    
    # Initialize source
    source = Source(processing_mode='opencv',source_type=source_type, source_path=source_path)
    
    # Initialize model
    model_initializer = ModelInitializer(model_type=model_type,model_path=model_path)
    model = model_initializer.get_model()
    
    # Initialize visualizer
    visualizer = Visualizer()

    while source.cap.isOpened():
        frame = source.get_frame()
        if frame is None:
            break

        # Run inference
        results = model.predict(frame)
        
        # Display results
        visualizer.display_results(frame, results, task=model_type.split('_')[1], names=getattr(model.model, 'names', None))

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    source.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()



