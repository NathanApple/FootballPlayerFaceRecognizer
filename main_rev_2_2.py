import cv2 as cv
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Initialize lists for storing face images and class labels
face_list = []
class_list = []

# Load the face cascade classifier
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

# Set the path to the dataset
train_path = 'Dataset/dataset'
personName = os.listdir(train_path)

X_train = [] 
X_test = []
y_train = [] 
y_test = []

# Loop through each person's directory and their images
for idx, name in enumerate(personName):
  fullPath = train_path + '/' + name
    
  for img_name in os.listdir(fullPath):
    img_fullPath = fullPath + '/' + img_name
    img = cv.imread(img_fullPath, 0)  # Read image in BGR color mode
        
    detected_face = face_cascade.detectMultiScale(img, scaleFactor=1.15, minNeighbors=5)
        
    if len(detected_face) < 1:
      continue
        
    for face_rect in detected_face:
      x, y, h, w = face_rect
      face_img = img[y:y+h, x:x+w]
            
      # Histogram Equalization -> improve contrast of image
      face_img = cv.equalizeHist(face_img)
            
      # Apply Gaussian blur -> reduce noise
      face_img = cv.GaussianBlur(face_img, (5, 5), 0)
      
      face_img = cv.bilateralFilter(face_img, d=3, sigmaColor=200, sigmaSpace=200)
      
      # Resize the image to a specific size (e.g., 100x100)
      face_img = cv.resize(face_img, (75, 75))
            
      face_list.append(face_img)
      class_list.append(idx)

# Convert class_list to a NumPy array for training
class_list = np.array(class_list)

print("size of faces list : ", +len(face_list))

# Split the data into training and testing sets (e.g., 75% train, 25% test)
X_train, X_test, y_train, y_test = train_test_split(face_list, class_list, test_size=0.25, random_state=42, stratify=class_list)

# Create LBPH Face Recognizer
face_recognizer = cv.face_LBPHFaceRecognizer.create()

# Train the recognizer on the training data
face_recognizer.train(X_train, y_train)

# Initialize a list to store the predicted labels
predicted_labels = []

# Iterate through the test set and predict labels
for test_img in X_test:
    predicted_label, _ = face_recognizer.predict(test_img)
    predicted_labels.append(predicted_label)

# Calculate accuracy
accuracy = accuracy_score(y_test, predicted_labels)
print(f"Accuracy: {accuracy * 100:.2f}%")



test_path = input("Enter the path of the test image : ")

img_bgr = cv.imread(test_path)

detected_face = face_cascade.detectMultiScale(img_bgr,scaleFactor=1.15,minNeighbors=5)
  
if len(detected_face) < 1:
  print("No Face Detected")
else:
  for face_rect in detected_face:
    x,y,h,w = face_rect
    face_img = img_bgr[y:y+h, x:x+w]
    # Resize the image to a specific size (e.g., 100x100)
    face_img = cv.resize(face_img, (75, 75))
    print("resize test")
    # Convert to grayscale 
    face_img = cv.cvtColor(face_img, cv.COLOR_BGR2GRAY)
    print("cvt test")
            
    # Histogram Equalization -> improve contrast of image
    face_img = cv.equalizeHist(face_img)
    print("equalize hist test")

    # Apply Gaussian blur -> reduce noise
    face_img = cv.GaussianBlur(face_img, (5, 5), 0)
    print("gauss blur")
    
    res, confidence = face_recognizer.predict(face_img)
    print("Predicted Label: " + str(res))
    print("Confidence Level: " + str(confidence))
    cv.rectangle(img_bgr, (x,y), (x+w,y+h), (255,0,0), 1)
    text = personName[res] + ' : ' + str(confidence)
    cv.putText(img_bgr, text, (x,y-10), cv.FONT_HERSHEY_PLAIN, 1.5, (0,255,0), 2)
    cv.imshow('Result',img_bgr)
    cv.waitKey(0)