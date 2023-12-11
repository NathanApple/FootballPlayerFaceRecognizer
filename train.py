import cv2
import os
import numpy as np
import random

def preprocess_image(face_img):
    face_img = cv2.resize(face_img, (75, 75))

    # Convert to grayscale
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

    # Histogram Equalization -> improve contrast of image
    face_img = cv2.equalizeHist(face_img)

    # Apply Gaussian blur -> reduce noise
    face_img = cv2.GaussianBlur(face_img, (5, 5), 0)
    
    return face_img

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
    # full_path = name

    for img_full_path in person_list:
        img = cv2.imread(img_full_path, 0)

        detected_face = face_cascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=5)

        if len(detected_face) < 1:
            continue
        
        if len(detected_face) > 1:
            extra_face += len(detected_face) - 1
            # img_bgr = cv2.imread(img_full_path)
            # for face_rect in detected_face:
            #     x, y, h, w = face_rect
            #     cv2.rectangle(img_bgr,  (x, y), (x+w, y+h), (0, 255, 0))
            # cv2.imshow('Result '+img_full_path, img_bgr)
            # cv2.waitKey(0)
            continue;
            pass;
        
        for face_rect in detected_face:
            x, y, h, w = face_rect
            face_img = img[y:y*h, x:x+w]

            face_rect = preprocess_image(face_rect)

            face_list.append(face_img)
            class_list.append(idx)



# print("Total Face Detected: " + str(len(class_list)))
# print(len(face_list))
print("Total Extra Face Detected: " + str(extra_face))

# exit()


face_recognizer = cv2.face.LBPHFaceRecognizer_create()
print(np.array(class_list))
face_recognizer.train(face_list, np.array(class_list))

test_path = 'Dataset/example/test'

accuracy_list = []

for idx, person_list in enumerate(test_list):
    # print(test_list)
    for img_full_path in person_list:
        # print(img_full_path)
        img_gray = cv2.imread(img_full_path, 0)
        img_bgr = cv2.imread(img_full_path)

        detected_face = face_cascade.detectMultiScale(img_bgr, scaleFactor=1.2, minNeighbors=5)

        if len(detected_face) < 1:
            accuracy_list.append(False)
            continue
        
        if len(detected_face) > 1:
            accuracy_list.append(False)
            # extra_face += len(detected_face) - 1
            # img_bgr = cv2.imread(img_full_path)
            # for face_rect in detected_face:
            #     x, y, h, w = face_rect
            #     cv2.rectangle(img_bgr,  (x, y), (x+w, y+h), (0, 255, 0))
            # cv2.imshow('Result '+img_full_path, img_bgr)
            # cv2.waitKey(0)
            continue;
        
        for face_rect in detected_face:
            x, y, h, w = face_rect
            face_img = img_gray[y:y+h, x:x+w]
            
            face_rect = preprocess_image(face_rect)
            
            res, confidence = face_recognizer.predict(face_img)
            # print(person_name[res])
            cv2.rectangle(img_bgr,  (x, y), (x+w, y+h), (0, 255, 0))
            text = person_name[res] + ' : ' + str(confidence)
            # cv2.putText(img_bgr, text, (x, y-10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
            # cv2.imshow('Result '+str(person_name[idx]) , img_bgr)
            # cv2.waitKey(0)
            if (person_name[idx] == person_name[res]):
                accuracy_list.append(True)
            else:
                accuracy_list.append(False)
            # exit()

print(str(sum(accuracy_list)/len(accuracy_list)*100) + '%')
        # if ()
        
