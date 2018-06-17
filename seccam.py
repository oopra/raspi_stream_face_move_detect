import io
import picamera
import logging
import socketserver
import threading
from http import server
import numpy as np
import warnings
import datetime
import imutils
import json
import cv2
from PIL import Image
import argparse
import sys, os
import socket
import cgi
import time

PAGE="""\
<html>
<head>
<style>
.imgContainer{
    float:left;
}
</style>
<title>Security Camera</title>
</head>
<body>
<div class="row"> 
<div class="imgContainer">
<h1>Live Streaming</h1>
<img src="stream.mjpg" width="320" height="240" />
</div>
<div class="imgContainer">
<h1>Last Face Detected</h1>
<img src="lastface.jpg" width="320" height="240" />
  <form action="http://@HOST@:8000/button/face" method="post">
   <button type='submit' name='face' value=-1><</button>
   <button type='submit' name='face' value=1>></button>
  </form>
</div>
<div class="imgContainer">
<h1>Last movement Detected</h1>
<img src="movement.jpg" width="320" height="240" />
  <form action="http://@HOST@:8000/button/move" method="post">
   <button type='submit' name='move' value=-1><</button>
   <button type='submit' name='move' value=1>></button>
  </form>
</div>
</div>
</body>
</html>
"""
dir_path = os.path.dirname(os.path.realpath(__file__))

if not os.path.exists(dir_path + '/face'):
    os.makedirs(dir_path + '/face')
if not os.path.exists(dir_path + '/move'):
    os.makedirs(dir_path + '/move')
faceidx = len([name for name in os.listdir(dir_path + '/face/')]) - 1
moveidx = len([name for name in os.listdir(dir_path + '/move/')]) - 1
face_cascade = cv2.CascadeClassifier(dir_path + '/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(dir_path + '/haarcascade_eye.xml')

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True,help="path to the JSON configuration file")
args = vars(ap.parse_args())

# filter warnings, load the configuration
# client
warnings.filterwarnings("ignore")
conf = json.load(open(args["conf"]))
client = None
MAX_RECS = conf["max_snapshots"]

def get_ip_address():
    """ get ip address of active network interface """

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    addr = s.getsockname()[0]
    s.close()
    return addr

PAGE = PAGE.replace('@HOST@', get_ip_address())

class LiveStreaming(threading.Thread):
    def __init__(self, owner):
        super(LiveStreaming, self).__init__()
        self.stream = io.BytesIO()
        self.bufwriteevent = threading.Event()
        self.framewriteevent = threading.Event()
        self.terminated = False
        self.owner = owner
        self.frame = None
        self.start()
    def run(self):
        # This method runs in a separate thread
        while not self.terminated:
            # Wait for an image to be written to the stream
            if self.bufwriteevent.wait():
                try:
                    self.stream.seek(0)
                    self.frame = self.stream.getvalue()
                    self.framewriteevent.set()
                except Exception as e:
                    logging.warning(
                        'LiveStreaming error: %s', str(e))
                finally:
                    # Reset the stream and bufwriteevent
                    self.stream.seek(0)
                    self.stream.truncate()
                    self.bufwriteevent.clear()

class LastFaceDetected(threading.Thread):
    def __init__(self, owner):
        super(LastFaceDetected, self).__init__()
        #self.stream = io.BytesIO()
        self.gray = None
        self.bufwriteevent = threading.Event()
        self.terminated = False
        self.owner = owner
        self.frame = None
        self.start()

    def run(self):
        count = 0
        # This method runs in a separate thread
        while not self.terminated:
            time.sleep(conf["camera_wait_time"])
            # Wait for an image to be written to the stream
            if self.bufwriteevent.wait():
                try:
                    frame  = self.frame
                    gray = self.gray
                    faces = face_cascade.detectMultiScale(
                        gray,
                        scaleFactor=1.1,
                        minNeighbors=5,
                        minSize=(30, 30),
                        flags = cv2.CASCADE_SCALE_IMAGE
                    )
                    # Draw a rectangle around the faces
                    for (x, y, w, h) in faces:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    if len(faces):
                        cv2.imwrite(dir_path + '/face/lastface_' + str(count) + '.jpg', frame)
                        count += 1
                        if count >= MAX_RECS:
                            count = 0
                except Exception as e:
                    logging.warning(
                        'LastFaceDetected error: %s', str(e))
                finally:
                    self.bufwriteevent.clear()


class LastMovementDetected(threading.Thread):
    def __init__(self, owner):
        super(LastMovementDetected, self).__init__()
        self.stream = io.BytesIO()
        self.bufwriteevent = threading.Event()
        self.terminated = False
        self.owner = owner
        self.frame = None
        self.avg = None
        self.start()
    def run(self):
        count = 0
        time.sleep(conf["camera_wait_time"])
        # This method runs in a separate thread
        while not self.terminated:
            # Wait for an image to be written to the stream
            if self.bufwriteevent.wait():
                try:
                    while True :
                        self.stream.seek(0)
                        frame = np.asarray(Image.open(self.stream))
                        framebu = frame.copy()
                        timestamp = datetime.datetime.now()
                        text = "Unoccupied"
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        graybu = gray.copy()
                        gray = cv2.GaussianBlur(gray, (21, 21), 0)
                        # if the average frame is None, initialize it
                        if self.avg is None:
                            print("[INFO] starting background model...")
                            self.avg = gray.copy().astype("float")
                            break
                        # accumulate the weighted average between the current frame and
                        # previous frames, then compute the difference between the current
                        # frame and running average
                        cv2.accumulateWeighted(gray, self.avg, 0.5)
                        frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(self.avg))
                        # threshold the delta image, dilate the thresholded image to fill
                        # in holes, then find contours on thresholded image
                        thresh = cv2.threshold(frameDelta, conf["delta_thresh"], 255,
                                               cv2.THRESH_BINARY)[1]
                        thresh = cv2.dilate(thresh, None, iterations=2)
                        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                                cv2.CHAIN_APPROX_SIMPLE)
                        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
                        # loop over the contours
                        for c in cnts:
                            # if the contour is too small, ignore it
                            if cv2.contourArea(c) < conf["min_area"]:
                                continue
                            # compute the bounding box for the contour, draw it on the frame
                            # and update the text
                            (x, y, w, h) = cv2.boundingRect(c)
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            text = "Occupied"
                        # draw the text and timestamp on the frame
                        ts = timestamp.strftime("%A %d %B %Y %I:%M:%S%p")
                        cv2.putText(frame, "Room Status: {}".format(text), (10, 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        
                        cv2.putText(frame, ts, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.35, (0, 0, 255), 1)
                        if text == "Occupied":
                            cv2.imwrite(dir_path + '/move/move_' + str(count) + '.jpg', frame)
                            count += 1
                            if count >= MAX_RECS:
                                count = 0
                            if not self.owner.faceDetection.bufwriteevent.is_set() :
                                self.owner.faceDetection.frame = framebu
                                self.owner.faceDetection.gray = graybu
                                self.owner.faceDetection.bufwriteevent.set()
                        break
                except Exception as e:
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    print(exc_type, fname, exc_tb.tb_lineno)
                    logging.warning(
                        'LastMovementDetected error: %s', str(e))
                finally:
                    # Reset the stream and bufwriteevent
                    self.stream.seek(0)
                    self.stream.truncate()
                    self.bufwriteevent.clear()


class ProcessOutput(object):
    def __init__(self):
        self.liveStreaming = LiveStreaming(self)
        self.faceDetection = LastFaceDetected(self)
        self.movementDetection = LastMovementDetected(self)
        self.currMaxTemp = 0
        
    def write(self, buf):
        if buf.startswith(b'\xff\xd8'):
            currTemp = int(open('/sys/class/thermal/thermal_zone0/temp').read()) / 1e3
            if currTemp > self.currMaxTemp:
                self.currMaxTemp = currTemp
                print("Maximum Temperature :", self.currMaxTemp)
            # New frame; set the current processor going and grab
            if not self.movementDetection.bufwriteevent.is_set() :
                self.movementDetection.stream.write(buf)
                self.movementDetection.bufwriteevent.set()
            if not self.liveStreaming.bufwriteevent.is_set() :
                self.liveStreaming.stream.write(buf)
                self.liveStreaming.bufwriteevent.set()



    def flush(self):
        # When told to flush (this indicates end of recording), shut
        # down in an orderly fashion. 
        self.liveStreaming.terminated = True
        self.faceDetection.terminated = True
        self.movementDetection.terminated = True

class StreamingHandler(server.BaseHTTPRequestHandler):
    def do_POST(self):
        # get the form data
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={'REQUEST_METHOD': 'POST',
                     'CONTENT_TYPE': self.headers['Content-Type'], })
        # check path
        if self.path.endswith('/button/face'):
            self.send_response(301)
            self.send_header('Location', '/index.html')
            self.end_headers()
            global faceidx
            faceidx += int(form["face"].value)
            totfiles = len([name for name in os.listdir(dir_path + '/face/')]) - 1
            
            if faceidx < 0:
                faceidx = len([name for name in os.listdir(dir_path + '/face/')]) - 1
            if faceidx > totfiles:
                faceidx = 0                
                
            #print("face %s" % faceidx)
        elif self.path.endswith('/button/move'):
            self.send_response(301)
            self.send_header('Location', '/index.html')
            self.end_headers()
            global moveidx
            moveidx += int(form["move"].value)
            totfiles = len([name for name in os.listdir(dir_path + '/move/')]) - 1

            if moveidx < 0:
                moveidx = totfiles
            if moveidx > totfiles:
                moveidx = 0
                
            #print("move %s" % moveidx)            
        
            
    def do_GET(self):
        if self.path == '/':
            self.send_response(301)
            self.send_header('Location', '/index.html')
            self.end_headers()
        elif self.path == '/index.html':
            content = PAGE.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
        elif self.path == '/movement.jpg':
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            global moveidx
            try:
                if moveidx < 0:
                    return
                with open(dir_path + '/move/move_' + str(moveidx) + '.jpg', 'rb') as f:
                    contents = f.read()
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', len(contents))
                    self.end_headers()
                    self.wfile.write(contents)
                    f.close()
            except Exception as e:
                logging.warning(
                    'MovementDetection : Removed streaming client %s: %s',
                    self.client_address, str(e))
        elif self.path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:
                while True:
                    if output.liveStreaming:
                        if output.liveStreaming.framewriteevent.wait():
                            if hasattr(output.liveStreaming, 'frame'):
                                frame = output.liveStreaming.frame
                                self.wfile.write(b'--FRAME\r\n')
                                self.send_header('Content-Type', 'image/jpeg')
                                self.send_header('Content-Length', len(frame))
                                self.end_headers()
                                self.wfile.write(frame)
                                output.liveStreaming.framewriteevent.clear()
                                self.wfile.write(b'\r\n')                                
            except Exception as e:
                logging.warning(
                    'LiveStreaming : Removed streaming client %s: %s',
                    self.client_address, str(e))
        elif self.path == '/lastface.jpg':
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            global faceidx
            try:
                if faceidx < 0 :
                    return
                with open(dir_path + '/face/lastface_' + str(faceidx) + '.jpg', 'rb') as f:
                    contents = f.read()
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', len(contents))
                    self.end_headers()
                    self.wfile.write(contents)
                    f.close()
            except Exception as e:
                logging.warning(
                    'FaceDetection : Removed streaming client %s: %s',
                    self.client_address, str(e))             
        else:
            self.send_error(404)
            self.end_headers()

class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True

with picamera.PiCamera(resolution=tuple(conf["resolution"]), framerate=conf["fps"]) as camera:
    output = ProcessOutput()
    camera.vflip = True
    camera.hflip = True

    camera.start_recording(output, format='mjpeg')
    try:
        address = ('', 8000)
        server = StreamingServer(address, StreamingHandler)
        server.serve_forever()
    finally:
        camera.stop_recording()
