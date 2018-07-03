from collections import namedtuple
import cv2
import imutils
import os
import pickle

FaceRecognizerInstance = namedtuple("FaceRecognizerInstance", ["trained", "labels"])

class FaceRecognizer:
	def __init__(self, recognizer, trained=False, labels=None):
		self.recognizer = recognizer
		self.trained = trained
		self.labels = labels

	def setLabels(self, labels):
		self.labels = labels

	def setConfidenceThreshold(self, confidenceThreshold):
		if imutils.is_cv2():
			self.recognizer.setDouble("threshold", confidenceThreshold)
		else:
			self.recognizer.setThreshold(confidenceThreshold)

	def train(self, data, labels):
		if not self.trained:
			self.recognizer.train(data, labels)
			self.trained = True
			return

		self.recognizer.update(data, labels)

	def predict(self, face):
		(prediction, confidence) = self.recognizer.predict(face)

		if prediction == -1:
			return ("Unknown", 0)

		return (self.labels[prediction], confidence)

	def save(self, basePath):
		# construct the face recognizer instance
		fri = FaceRecognizerInstance(trained=self.trained, labels=self.labels)
 
		# due to strange behavior with OpenCV, we need to make sure the output classifier file
		# exists prior to writing it to file
		if not os.path.exists(basePath + "/classifier.model"):
			f = open(basePath + "/classifier.model", "w")
			f.close()
 
		# handle if we're writing the OpenCV 2.4 recognizer to file
		if imutils.is_cv2():
			self.recognizer.save(basePath + "/classifier.model")
 
		# otherwise, we're writing the OpenCV 3 recognizer to file
		else:
			self.recognizer.write(basePath + "/classifier.model")
 
		# write the parameters to file
		f = open(basePath + "/fr.cpickle", "wb")
		f.write(pickle.dumps(fri))
		f.close()
 
	@staticmethod
	def load(basePath):
		# load the face recognition instance and construct the OpenCV face recognizer
		fri = pickle.loads(open(basePath + "/fr.cpickle", "rb").read())
 
		# handle if we are building an OpenCV 2.4 face recognizer
		if imutils.is_cv2():
			recognizer = cv2.createLBPHFaceRecognizer()
			recognizer.load(basePath + "/classifier.model")
 
		# otherwise we are building an OpenCV 3 face recognizer
		else:
			recognizer = cv2.face.LBPHFaceRecognizer_create()
			recognizer.read(basePath + "/classifier.model")
 
		# construct and return the face recognizer
		return FaceRecognizer(recognizer, trained=fri.trained, labels=fri.labels)