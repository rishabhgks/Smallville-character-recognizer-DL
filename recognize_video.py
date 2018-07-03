# USAGE
# python video_template.py
from pyimagesearch.face_recognition import FaceDetector
from pyimagesearch.face_recognition import FaceRecognizer
import argparse
import imutils
import cv2
 
# construct the argument parse and parse command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face-cascade", required=True, help="path to face detection cascade")
ap.add_argument("-c", "--classifier", required=True, help="path to the classifier")
ap.add_argument("-t", "--confidence", type=float, default=100.0,
	help="maximum confidence threshold for positive face identification")
ap.add_argument("-v", "--video", help="path to the (optional) video file")
args = vars(ap.parse_args())
 
# initialize the face detector, load the face recognizer, and set the confidence
# threshold
fd = FaceDetector(args["face_cascade"])
fr = FaceRecognizer.load(args["classifier"])
fr.setConfidenceThreshold(args["confidence"])

# if a video path was not supplied, grab the reference to the webcam
if not args.get("video", False):
	camera = cv2.VideoCapture(0)

# otherwise, grab a reference to the video file
else:
	camera = cv2.VideoCapture(args["video"])

# keep looping
while True:
	# grab the current frame
	(grabbed, frame) = camera.read()

	# if we are viewing a video and we did not grab a
	# frame, then we have reached the end of the video
	if args.get("video") and not grabbed:
		break

	# show the frame to our screen
	cv2.imshow("Frame", imutils.resize(frame, width=500))
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faceRects = fd.detect(gray, scaleFactor=1.06, minNeighbors=7, minSize=(60, 60))
	# loop over the face bounding boxes
	for (i, (x, y, w, h)) in enumerate(faceRects):
		# grab the face to predict
		face = gray[y:y + h, x:x + w]
 
		# predict who's face it is, display the text on the image, and draw a bounding
		# box around the face
		(prediction, confidence) = fr.predict(face)
		prediction = "{}: {:.2f}".format(prediction, confidence)
		cv2.putText(frame, prediction, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
 
	# show the frame and record if the user presses a key
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()