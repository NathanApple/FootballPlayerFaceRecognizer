import cv2
import os
import numpy as np
import random

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

face_list = []
class_list = []

dataset_path = 'Dataset/dataset'

person_name = os.listdir(dataset_path)

train_list = []
test_list = []

for idx, name in enumerate(person_name):
    full_path = dataset_path + '/' + name
    image_name = os.listdir(full_path)
    
    image_list = [full_path + "/" + image for image in image_name]
    
    random.shuffle(image_list)

    train_data = image_list[:15]
    train_list.append(train_data)
    
    test_data = image_list[15:]
    test_list.append(test_data)

# print(len(train_list))
# print(len(test_list))

# print(test_list)

# for idx, name in enumerate(train_list):

# exit()

# person_name = os.listdir(train_path)

extra_face = 0

for idx, person_list in enumerate(train_list):
    full_path = name

    for img_full_path in person_list:
        img = cv2.imread(img_full_path, 0)

        detected_face = face_cascade.detectMultiScale(img, scaleFactor=1.5, minNeighbors=7)

        if len(detected_face) < 1:
            continue
        
        if len(detected_face) > 1:
            extra_face += len(detected_face) - 1
            img_bgr = cv2.imread(img_full_path)
            for face_rect in detected_face:
                x, y, h, w = face_rect
                cv2.rectangle(img_bgr,  (x, y), (x+w, y+h), (0, 255, 0))
            cv2.imshow('Result '+img_full_path, img_bgr)
            cv2.waitKey(0)
        
        for face_rect in detected_face:
            x, y, h, w = face_rect
            face_img = img[y:y*h, x:x+w]
            
            face_list.append(face_img)
            class_list.append(idx)



print("Total Face Detected: " + str(len(class_list)))
# print(len(face_list))
print("Total Extra Face Detected: " + str(extra_face))

exit()


face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(face_list, np.array(class_list))

test_path = 'Dataset/example/test'

accuracy_list = []

for image_name in os.listdir(test_path):
    full_img_path = test_path + "/" + image_name
    img_gray = cv2.imread(full_img_path, 0)
    img_bgr = cv2.imread(full_img_path)
    
    detected_face = face_cascade.detectMultiScale(img_gray, scaleFactor=1.2, minNeighbors=11)
    
    if len(detected_face) < 1:
        continue
    
    for face_rect in detected_face:
        x, y, h, w = face_rect
        face_img = img_gray[y:y+h, x:x+w]
        
        res, confidence = face_recognizer.predict(face_img)
        
        cv2.rectangle(img_bgr,  (x, y), (x+w, y+h), (0, 255, 0))
        text = person_name[res] + ' : ' + str(confidence)
        cv2.putText(img_bgr, text, (x, y-10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
        cv2.imshow('Result '+image_name, img_bgr)
        cv2.waitKey(0)
        
