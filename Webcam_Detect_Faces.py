#!/usr/local/bin/python3

# -*- coding: utf-8 -*-

# Setup: pip3 install opencv-python face_recognition

import cv2, face_recognition, re, pickle
from face_recognition.face_recognition_cli import image_files_in_folder

work_from_cache = 0

known_faces = []
known_face_names = []
face_locations = []
counter_faces = 0
counter_total = 0

if(work_from_cache):
	with open('known_faces.dmp', 'rb') as f:
		known_faces = pickle.load(f)
		known_face_names = pickle.load(f)
else:
	for img_path in image_files_in_folder('photos'):
		counter_total += 1
		try:
			face_image = face_recognition.load_image_file(img_path)
			face_encoding = face_recognition.face_encodings(face_image)[0]
			known_faces.append(face_encoding)
			name = re.sub(r'(.*)/(.*?)\.(.*)',r'\2',img_path)
			counter_faces += 1
			print(counter_faces, '/', counter_total, ' ID: ', name)
			known_face_names.append(name)
		except:
			print('Face not found in ', img_path)
	with open('known_faces.dmp', 'wb') as f:
		pickle.dump(known_faces, f)
		pickle.dump(known_face_names, f)

video_capture = cv2.VideoCapture(0)

while True:
	ret, frame = video_capture.read()
	rgb_frame = frame[:, :, ::-1]	
	face_locations = face_recognition.face_locations(rgb_frame)
	face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
	face_names = []
	for face_encoding in face_encodings:
		matches = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.50)
		name = "Unknown"
		if True in matches:
			first_match_index = matches.index(True)
			name = known_face_names[first_match_index]
		face_names.append(name)
	for (top, right, bottom, left), name in zip(face_locations, face_names):
		if not name:
			continue
		cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
		cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
		font = cv2.FONT_HERSHEY_DUPLEX
		cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
	cv2.imshow('Video', frame)	
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
video_capture.release()
cv2.destroyAllWindows()
