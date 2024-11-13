import cv2
import numpy as np

class Visualizer:
    def visualize_detection(self, frame, boxes, names):
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            class_id = int(box.cls.item())          # Class ID
            confidence = box.conf.item()            # Confidence score
            label = f"{names[class_id]}: {confidence:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    def visualize_segmentation(self, frame, results):
        if results.masks is None:
            print("No segmentation masks detected.")
            return frame  # Return the frame without any mask overlay

        try:
            # Assuming results.masks is a tensor or list of masks
            mask = results.masks.data[0].cpu().numpy()  # Get the first mask
            mask = (mask * 255).astype(np.uint8)  # Scale mask to 0-255
            color_mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)

            # Resize color mask to match the frame's size
            color_mask = cv2.resize(color_mask, (frame.shape[1], frame.shape[0]))

            # Blend the color mask with the original frame
            frame = cv2.addWeighted(frame, 0.7, color_mask, 0.3, 0)
        except AttributeError as e:
            print(f"Error processing segmentation masks: {e}")
        
        return frame


    def visualize_classification(self, frame, results):
        """
        Visualize the top classification result on the frame.

        Args:
            frame (np.array): The image frame to display classification results on.
            results: YOLO model results object containing classification probabilities.
        """
        # Access the Probs object
        probs = results.probs

        # Get the predicted class id and its confidence
        top_class_id = probs.top1  # Index of the class with the highest probability
        predicted_class = results.names[int(top_class_id)]  # Get the class name
        confidence = probs.top1conf.item() if probs.top1conf is not None else 0  # Confidence score for the top class

        # Display the classification result on the frame
        label = f"{predicted_class}: {confidence:.2f}"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        return frame






    def display_results(self, frame, results, task='detection', names=None):
        if task == 'detection':
            self.visualize_detection(frame, results.boxes, names)
        elif task == 'segmentation':
            frame = self.visualize_segmentation(frame, results)  # Pass results directly
        elif task == 'classification':
            self.visualize_classification(frame, results)
        else:
            print("Unsupported task type for visualization")

        # cv2.imshow('Results', frame)
        # cv2.waitKey(1)  # Ensure real-time display update
