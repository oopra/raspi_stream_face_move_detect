# raspi_stream_face_move_detect
Live streaming, movement detection and face detection using Raspberry Pi camera. It uses the PiCamera python module to access the camera connected to the Raspberry Pi. The Raspberry Pi is connected to the network and the camera recording is viewed with a browser at : http://<RaspberryPiIPaddress>:8000/index.html. The configuration is input via conf.json.
 ## Command
python3 seccam.py --conf conf.json
## Introduction
This python script does the following:
1. Live stream the camera
2. Detect movements and capture
3. Detect faces and capture
## Thanks
I took references from the following webpages
1. https://www.pyimagesearch.com
2. https://realpython.com/
