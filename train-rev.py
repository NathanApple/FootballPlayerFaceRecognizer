import cv2 as cv
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def main():
    while True:
        choice = showMenu()
        if choice == "1":
            print("Training and Testing")
            train()
        elif choice == "2":
            predict()
        elif choice == "3":
            exit()

def showMenu():
    print("Football Player Face Recognition")
    print("1. Train and Test Model")
    print("2. Predict")
    print("3. Exit")
    choice = input(">> ")
    return choice



def preprocess_image(face_img):
    face_img = cv.resize(face_img, (75, 75))

    # Convert to grayscale
    face_img = cv.cvtColor(face_img, cv.COLOR_BGR2GRAY)

    # Histogram Equalization -> improve contrast of image
    face_img = cv.equalizeHist(face_img)

    # Apply Gaussian blur -> reduce noise
    face_img = cv.GaussianBlur(face_img, (5, 5), 0)
    face_img = cv.bilateralFilter(face_img, d=3, sigmaColor=200, sigmaSpace=200)
    
    return face_img

# Initialize lists for storing face images and class labels
face_list = []
class_list = []

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
face_recognizer = cv.face_LBPHFaceRecognizer.create()

# Set the path to the dataset
train_path = 'Dataset/dataset'
personName = os.listdir(train_path)

def train():
    X_train = []
    X_test = []
    y_train = []
    y_test = []

    # Loop through each person's directory and their images
    for idx, name in enumerate(personName):
        fullPath = train_path + '/' + name

        for img_name in os.listdir(fullPath):
            img_fullPath = fullPath + '/' + img_name
            img = cv.imread(img_fullPath)

            detected_face = face_cascade.detectMultiScale(img, scaleFactor=1.15, minNeighbors=5)

            if len(detected_face) < 1:
                continue

            for face_rect in detected_face:
                x, y, h, w = face_rect
                face_img = img[y:y+h, x:x+w]

                face_img = preprocess_image(face_img)

                face_list.append(face_img)
                class_list.append(idx)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(face_list, np.array(class_list),
                                                        test_size=0.25, random_state=99,
                                                        stratify=class_list)

    # print(len(X_train), len(X_test))

    # Train
    face_recognizer.train(X_train, y_train)

    predicted_labels = []

    for test_img in X_test:
        predicted_label, _ = face_recognizer.predict(test_img)
        predicted_labels.append(predicted_label)

    # Accuracy
    accuracy = accuracy_score(y_test, predicted_labels)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("Training and Testing finished")
    input("Press enter to continue...")
    


def predict():
    test_path = input("Input absolute path for the image : ")

    if os.path.isfile(test_path) == False:
        print("File not found")
        return

    img_bgr = cv.imread(test_path)

    detected_face = face_cascade.detectMultiScale(img_bgr,scaleFactor=1.15,minNeighbors=5)
    
    if len(detected_face) < 1:
        print("No Face Detected")
    else:
        for face_rect in detected_face:
            x,y,h,w = face_rect
            face_img = img_bgr[y:y+h, x:x+w]

            face_img = preprocess_image(face_img)

            res, confidence = face_recognizer.predict(face_img)
            print("Predicted Label: " + str(res))
            if (confidence == 0):
                print("Confidence Level: " + str(confidence) + ", Perfect Match!")
            else:
                print("Confidence Level: " + str(confidence))
            cv.rectangle(img_bgr, (x,y), (x+w,y+h), (0,255,0), 1)
            text = personName[res] + ' : ' + str(confidence)
            cv.putText(img_bgr, text, (x,y-10), cv.FONT_HERSHEY_PLAIN, 1.5, (0,255,0), 2)
            cv.imshow('Result',img_bgr)
            cv.waitKey(0)

if __name__ == '__main__':
    main()
