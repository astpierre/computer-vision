import os
import sys
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import face_recognition


def main():
    cap = cv.VideoCapture(1)

    # Load a sample picture and learn how to recognize it.
    andrew_image = face_recognition.load_image_file("knownFaces/andrew.jpg")
    andrew_face_encoding = face_recognition.face_encodings(andrew_image)[0]

    amy_image = face_recognition.load_image_file("knownFaces/amy.jpg")
    amy_face_encoding = face_recognition.face_encodings(amy_image)[0]

    # Create arrays of known face encodings and their names
    known_face_encodings = [
        andrew_face_encoding,
        amy_face_encoding
    ]
    known_face_names = [
        "Andrew St. Pierre",
        "Amy St. Pierre"
    ]

    while(True):
        ret, frame = cap.read()
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_frame = frame[:, :, ::-1]
        # Find all the faces and face enqcodings in the frame of video
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Loop through each face in this frame of video
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            # Draw a box around the face
            cv.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv.FILLED)
            font = cv.FONT_HERSHEY_DUPLEX
            cv.putText(frame, name, (left + 3, bottom - 6), font, 0.28, (255, 255, 255), 1)

        cv.imshow('Video', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
