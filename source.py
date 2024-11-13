import cv2
from yt_dlp import YoutubeDL


class Source:

    def __init__(self,processing_mode='opencv',source_type='webcam',source_path=None):
        self.processing_mode=processing_mode
        self.source_type=source_type
        self.source_path=source_path
        self.cap=None
        self.__initialize_processing_mode()

    def __initialize_processing_mode(self):
        if self.processing_mode=='opencv':
            self.__initialize_opencv()
        elif self.processing_mode=='ffmpeg':
            self.__initialize_ffmpeg()
        elif self.processing_mode=='gstreamer':
            self.__initialize_gstreamer()
        else:
            print("Invalid Processing Mode. Pls select a valid one!!!")

    def __initialize_opencv(self):
        if self.source_type=='webcam':
            self.cap=cv2.VideoCapture(self.source_path)
        elif self.source_type=='videofile':
            self.cap=cv2.VideoCapture(self.source_path)
        elif self.source_type=='url':
            self.cap=cv2.VideoCapture(self.source_path)
        elif self.source_type=='youtube':
            ydl_opts = {'format': 'best[ext=mp4]'}  # Fetch the best available mp4 format
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(self.source_path, download=False)
                url = info['url']
            self.cap=cv2.VideoCapture(url)
        else:
            raise ValueError("Unsupported source type. Use 'webcam', 'videofile', or 'youtube'.")

    def __initialize_ffmpeg(self):
        print("FFMPEG CODE TO BE WRITTEN!!!")
    
    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def release(self):
        self.cap.release()

