from modell import *
from helper import *
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
import keras
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from keras import backend as K 


nb_epoch = 25
pick = 3



num_classes = 4
batch_size = 8
save = pick + 1
# Reads paths of images together with their labels
filename = 'train_all.csv'
image_list, label_list = read_input_file(filename)
label_list = np.array(label_list)
label_list = np.transpose(label_list)

train_X, valid_X, train_label, valid_label = train_test_split(image_list, label_list, test_size=0.2, random_state=13, shuffle=True)

# print(train_X.shape[0])
# print(valid_X.shape)
# print(train_label.shape)
# print(valid_label.shape)


train_generator = generator(train_X, train_label, batch_size)
val_generator = generator(valid_X, valid_label, batch_size)

train_samples=np.ceil(train_X.shape[0] / batch_size)
val_samples=np.ceil(valid_X.shape[0] / batch_size)


# Load Model
json_file = open('model_'+str(pick)+'.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("model_"+str(pick)+".h5")
print("Loaded")
model.compile(loss=keras.losses.mse, optimizer=keras.optimizers.Adam(), metrics=['mse', bb_intersection_over_union])

# Train Model
# model = ResnetBuilder.build_resnet_9((3, 480, 640), 4)
model.fit_generator(train_generator, steps_per_epoch=train_samples, epochs=nb_epoch, verbose=1, 
                    validation_data=val_generator, validation_steps=val_samples)


# Save Model
model_json = model.to_json()
with open("model_"+str(save)+".json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_"+str(save)+".h5")
model.save("whole_model_"+str(save)+".h5")
print("Saved")

