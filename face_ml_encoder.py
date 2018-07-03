from __future__ import print_function
from imutils import paths
import pickle
import os
from pyimagesearch.face_recognition import FaceDetector
from imutils import encodings
import argparse
import imutils
import cv2

# construct the argument parse and parse command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", required=True, help="path to face detection cascade")
ap.add_argument("-o", "--output", required=True, help="path to output file")
ap.add_argument("-w", "--write-mode", type=str, default="a+", help="write method for the output file")
ap.add_argument("-i","--dataset", required=True, help = "path to input directory of faces + images")
args = vars(ap.parse_args())

fd = FaceDetector(args["face"])
f = open(args["output"], args["write_mode"])
total = 0
color = (0, 225, 0)

print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(args["dataset"]))
for (i, imagePath) in enumerate(imagePaths):
	image = cv2.imread(imagePath)
	frame = imutils.resize(image, width=500)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faceRects = fd.detect(gray, scaleFactor=1.05, minNeighbors=7, minSize=(60, 60))
	if len(faceRects) > 0:
		# sort the bounding boxes, keeping only the largest one
		(x, y, w, h) = max(faceRects, key=lambda b:(b[2] * b[3]))
		face = gray[y:y + h, x:x + w].copy(order="C")
		f.write("{}\n".format(encodings.base64_encode_image(face)))
		total+=1
		cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
	cv2.imshow("Frame", frame)
print("[INFO] wrote {} frames to file".format(total))
f.close()
cv2.destroyAllWindows()	
