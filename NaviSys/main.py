import cv2 as cv
import NewComparator
import Preprocessing
import SearchMethods
from sys import path
from time import time
import threading
import queue


def capture():
	pipeline = (
		'rtspsrc location="rtsp://192.168.144.25:8554/main.264" latency=1 '
		'! rtpjitterbuffer mode=1 do-lost=true latency=50 drop-on-latency=false '
		'! rtph264depay '
		'! h264parse '
		'! avdec_h264 '
		'! queue max-size-buffers=0 max-size-bytes=0 max-size-time=100000000 '
		'! videoconvert n-threads=2 '
		'! video/x-raw,format=BGR '  
		'! appsink sync=True'                  
	)
	cap = cv.VideoCapture(pipeline, cv.CAP_GSTREAMER)
	count_fails = 0
	while not cap.isOpened():
		print(f"Fail {count_fails + 1}. Try open stream again.")
		cap = cv.VideoCapture(pipeline, cv.CAP_GSTREAMER)
		count_fails += 1
		if count_fails >= 10:
			print("Fail more than 10 times. Please check connection.")
			exit()
		sleep(10)
	print("Capture OK\n")
	return cap



class Navigation:
    path_dir = path[0]
    path_main = path_dir + '/WK_00005-1.jpg'  # yand_maps_2-2.bmp MyMap-2.jpg WK_00005-1.jpg
    img1 = cv.imread(path_main, cv.IMREAD_GRAYSCALE)

    path_video = path_dir + "/move3.mp4"  # djelga_video2.mp4   test_move.mp4

    flight_altitude = 30  # текущая высота полета
    height_map = 500  # Высота съемки карты
    frame_queue = queue.Queue(maxsize=1)
    method_index = 1  # 1: "SIFT", 2: "AKAZE", 3: "ORB", 4: "ASIFT", 5: "SuperPoint"

    def __init__(self):
        cv.imwrite("main_with_points_new.jpg", self.img1)
        self.img1 = cv.GaussianBlur(self.img1, (5, 5), sigmaX=0, sigmaY=0)
        self.method = SearchMethods.Method(self.method_index, 0.48)
        self.kp1, self.des1 = self.method.get_kp_and_des(self.img1)
        #self.main_video_cycle()
        self.cap = capture()
        #self.cap = cv.VideoCapture(self.path_video)
        self.reader_thread = threading.Thread(target=self.frame_reader,
                            name="Get_frame",
                            args=(self.cap, self.frame_queue))
        self.processor_thread = threading.Thread(target=self.frame_processor, 
                            name="Main_process",
                            args=(self.frame_queue, 0))

        self.reader_thread.start()
        self.processor_thread.start()

        self.reader_thread.join()
        self.processor_thread.join()
        

    def frame_reader(self, cap, frame_queue):
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Очищаем очередь, чтобы всегда оставался только самый свежий кадр
            while not frame_queue.empty():
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    pass
            frame_queue.put(frame)
        cap.release()
    
    def frame_processor(self, frame_queue, interval=0):
        while True:
            try:
                frame = frame_queue.get(timeout=1)
            except queue.Empty:
                continue
            self.main_process(frame)
    
    def definition_of_blur(self, height, altitude):
        diff = int(height / altitude)
        if diff <= 5:
            return 5, 5
        elif diff % 2 == 1:
            return diff - 2, diff - 2
        else:
            return diff - 1, diff - 1

    def main_process(self, frame):
        # Предобработка кадра
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray = Preprocessing.resize_img(gray, 1024)
        kernel = self.definition_of_blur(self.height_map, self.flight_altitude)
        #kernel = (5, 5)
        gray = cv.GaussianBlur(gray, kernel, sigmaX=0, sigmaY=0)

        test = NewComparator.Compare(
            main_img=self.img1,
            kp_main=self.kp1,
            des_main=self.des1,
            height_main=self.height_map,
            img_2=gray,
            altitude=self.flight_altitude,
            method=self.method
        )

        print(test.comparator())


if __name__ == "__main__":
    test = Navigation()
