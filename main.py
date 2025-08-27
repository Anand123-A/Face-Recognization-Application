# Importing the required Libraries
import os
import cv2
import numpy as np

# ==========================
# Function to Collect Faces
# ==========================
def collects_faces():
    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Use DroidCam
    # If USB mode: cv2.VideoCapture(1)
    # If WiFi mode: cv2.VideoCapture("http://<YOUR_PHONE_IP>:4747/video")
    cap = cv2.VideoCapture(1)

    count = 0
    while True:
        ret, frame = cap.read()
        if ret == False:
            continue

        count += 1

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray_frame, 1.1, 4)

        for (x, y, w, h) in faces:
            captured_face = frame[y:y+h, x:x+w]

            cropped_face = cv2.resize(captured_face, (250, 250))
            gray_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2GRAY)

            faces_collection_directory = 'faces'
            if not os.path.exists(faces_collection_directory):
                os.mkdir(faces_collection_directory)

            face_path = os.path.join(faces_collection_directory, "user" + str(count) + ".jpg")
            cv2.imwrite(face_path, gray_face)

            cv2.putText(gray_face, str(count), (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow("Face", gray_face)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if count == 100:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"{count} number of Sample faces are Collected Successfully")

# ==========================
# Function to Train Model
# ==========================
def Training_model():
    faces_directory = 'faces'
    faces = []
    labels = []

    for i, filename in enumerate(os.listdir(faces_directory)):
        if filename.endswith(".jpg"):
            face_path = os.path.join(faces_directory, filename)
            face = cv2.imread(face_path)
            gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            faces.append(gray_face)
            labels.append(i)

    face_recognizer_model = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer_model.train(faces, np.array(labels))
    face_recognizer_model.save('trained_data.xml')

    print("Face recognizer model trained successfully")

# ==========================
# Function to Test Model
# ==========================
def Testing_model():
    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    face_recognizer_model = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer_model.read('trained_data.xml')

    # Use DroidCam here too
    cap = cv2.VideoCapture(1)

    while True:
        ret, frame = cap.read()
        if ret == False:
            continue

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray_frame, 1.1, 4)

        for (x, y, w, h) in faces:
            face_roi = gray_frame[y:y+h, x:x+w]
            label, confidence = face_recognizer_model.predict(face_roi)

            if confidence <= 85:
                cv2.putText(frame, "Known", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Unknown", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # press q to quit
            break

    cap.release()
    cv2.destroyAllWindows()

# ==========================
# Main Function
# ==========================
def main():
    while True:
        print("\nPress 1 to collect Sample Faces")
        print("Press 2 to Train the Model")
        print("Press 3 to Test the Model")
        print("Press 4 to Exit the App")

        choice = input("Enter Your choice :")

        if choice == '1':
            collects_faces()
        elif choice == '2':
            Training_model()
        elif choice == '3':
            Testing_model()
        elif choice == '4':
            break
        else:
            print("Invalid Choice Entered")

if __name__ == "__main__":
    main()
