import cv2 as cv
import RPi.GPIO as GPIO
import threading
import os
from datetime import datetime
from time import time
from time import sleep
from sys import path

if not os.path.exists(path[0]+f'/camera/imgs_{datetime.now().strftime("%Y-%m-%d")}') or not os.path.exists(path[0]+f'/camera/videos_{datetime.now().strftime("%Y-%m-%d")}'):
	os.mkdir(f'camera/imgs_{datetime.now().strftime("%Y-%m-%d")}')
	os.mkdir(f'camera/videos_{datetime.now().strftime("%Y-%m-%d")}')
	
cur = time()
TIME_BLINK = 1
SAVE_FOLDER_IMG = path[0] + f'/camera/imgs_{datetime.now().strftime("%Y-%m-%d")}'
SAVE_FOLDER_VIDEO = path[0] + f'/camera/videos_{datetime.now().strftime("%Y-%m-%d")}'
state = 0

GPIO.setmode(GPIO.BCM)
LED_GREEN = 23
LED_RED = 24
BTN_PHOTO = 13
BTN_STOP = 6
BTN_VIDEO = 26
BTN_POWER = 19

GPIO.setup([LED_GREEN, LED_RED], GPIO.OUT)
GPIO.setup([BTN_PHOTO, BTN_STOP, BTN_VIDEO, BTN_POWER], GPIO.IN, 
pull_up_down = GPIO.PUD_UP)

is_active = True
is_recording = False
is_photo_enable = True

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
	
def process_stream():
	global is_active, is_recording, is_photo_enable, cur, state
	cap = capture()
	video_writer = None
	
	while True:
		if is_active:
			ret, frame = cap.read()
			if ret:
				if is_recording:
					if video_writer is None:
						timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
						fourcc = cv.VideoWriter_fourcc(*'mp4v')
						video_writer = cv.VideoWriter(f'{SAVE_FOLDER_VIDEO}/VID_{timestamp}.mp4', 
						fourcc, 20.0, (1920, 1080))
					video_writer.write(frame)
					GPIO.output(LED_RED, GPIO.HIGH)
				elif is_photo_enable and (time() - cur > TIME_BLINK):
					cur = time()
					GPIO.output(LED_RED, state)
					state = not state
					print("new photo")
					timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
					cv.imwrite(f"{SAVE_FOLDER_IMG}/IMG_{timestamp}.jpg", frame)
					
					
					# sleep(TIME_BLINK)
					# GPIO.output(LED_RED, GPIO.LOW)
					# sleep(TIME_BLINK)
		else:
			GPIO.output(LED_RED, GPIO.LOW)
			if video_writer is not None:
				video_writer.release()
				video_writer = None
		
				
def button_callback(channel):
	global is_active, is_recording, is_photo_enable
	if channel == BTN_STOP:
		print("BTN STOP")
		is_active = False
		is_recording = False
		is_photo_enable = False
	elif channel == BTN_VIDEO:
		is_active = True
		is_recording = True
		is_photo_enable = False
	elif channel == BTN_PHOTO:
		print("BTN PHOTO")
		is_active = True
		is_photo_enable = True
		is_recording = False
	elif channel == BTN_POWER:
		GPIO.output(LED_GREEN, GPIO.HIGH)
		sleep(3)
		os.system("sudo shutdown now")
		

GPIO.add_event_detect(BTN_STOP, GPIO.RISING, callback=button_callback, 
	bouncetime=300)
GPIO.add_event_detect(BTN_VIDEO, GPIO.RISING, callback=button_callback, 
	bouncetime=300)
GPIO.add_event_detect(BTN_PHOTO, GPIO.RISING, callback=button_callback, 
	bouncetime=300)
GPIO.add_event_detect(BTN_POWER, GPIO.RISING, callback=button_callback, 
	bouncetime=300)
		
threading.Thread(target=process_stream, daemon=True).start()

try:
	while True:
		if not is_active:
			GPIO.output(LED_GREEN, GPIO.HIGH)
			sleep(TIME_BLINK)
			GPIO.output(LED_GREEN, GPIO.LOW)
			sleep(TIME_BLINK)
except KeyboardInterrupt:
	GPIO.cleanup()
