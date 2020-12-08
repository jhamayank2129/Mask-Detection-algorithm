import numpy as np
data = np.load('data.npy')
target = np.load('target.npy')


from keras.models import Sequential
from keras.layers import MaxPooling2D,Conv2D
from keras.layers import Dropout,Flatten,Dense,Activation
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split


model = Sequential()
model.add(Conv2D(200,(5,5),input_shape=data.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(100,(5,5)))
model.add(Activation('relu'))
model.add(MaxPooling2D(2,2))

model.add(Flatten())
##Flatten layer is used to stack out 2d convolutions to 1d

model.add(Dropout(0.5))
model.add(Dense(50,activation='relu'))
model.add(Dense(2,activation='softmax'))

model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')

train_data,test_data,train_target,test_target = train_test_split(data,target,test_size=0.1)

checkpoint = ModelCheckpoint('model-{epoch: }.model',monitor='val_loss',verbose=0,save_best_only=True,mode='auto')



history = model.fit(train_data,train_target,epochs = 20,validation_split=0.2,callbacks=[checkpoint])



