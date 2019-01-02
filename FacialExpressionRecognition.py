from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import  Flatten
from keras.layers import Dense
#classifier=Sequential()
#classifier.add(Conv2D(32,(1,3),input_shape=(182,182,3),activation='relu'))
## classifier.add(Conv2D(32,(3,3),input_shape=(64,64,3),activation='sigmoid'))
#classifier.add(MaxPooling2D(pool_size = (2, 2)))
#classifier.add(Conv2D(32, (1, 3), activation = 'relu'))
## classifier.add(Conv2D(32, (3, 3), activation = 'sigmoid'))
#classifier.add(MaxPooling2D(pool_size = (2, 2)))
#classifier.add(Flatten())
#classifier.add(Dense(units = 128, activation = 'relu'))
## classifier.add(Dense(units = 128, activation = 'sigmoid'))
#classifier.add(Dense(units = 1, activation = 'sigmoid'))
#classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
## classifier.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
#from keras.preprocessing.image import ImageDataGenerator
#train_datagen = ImageDataGenerator(rescale = 1./255,
#shear_range = 0.2,
#zoom_range = 0.2,
#horizontal_flip = True)
#test_datagen = ImageDataGenerator(rescale = 1./255)
#training_set = train_datagen.flow_from_directory('datasets/training',
#target_size = (182, 182),
#batch_size = 32,
#class_mode = 'binary')
#test_set = test_datagen.flow_from_directory('datasets/testing',
#target_size = (182, 182),
#batch_size = 32,
#class_mode = 'binary')
#classifier.fit_generator(training_set,
#steps_per_epoch = 50,
#epochs = 1,
#validation_data = test_set,
#validation_steps = 50)




import cv2
import numpy as np
from keras.preprocessing import image
cap=cv2.VideoCapture(0)
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
while(True):
    _,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    face=face_cascade.detectMultiScale(gray,1.5,5)
    for (x,y,w,h) in face:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    cv2.imshow("Face",frame)
    roi=frame[y:y+h,x:x+w]
    roi=resize(roi,(182,182,3))
    roi=np.expand_dims(roi,axis=0)
    res=classifier.predict(roi)
    if(res>=1):
        print("Happy")
    else:
        print("Sad")
    
    if(cv2.waitKey(1) & 0xFF==ord('q')):
        break
cap.release()
cv2.destroyAllWindows()
    