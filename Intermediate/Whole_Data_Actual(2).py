#!/usr/bin/env python
# coding: utf-8

# In[1]:


from modell import *
from keras.callbacks import ModelCheckpoint
import keras
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from keras import backend as K 

num_classes = 4
batch_size = 4

def read_input_file(filename):
    df = pd.read_csv(filename)
    l = df['image_name'].values
    x1, x2, y1, y2 = df['x1'].values, df['x2'].values, df['y1'].values, df['y2'].values
    return l, (x1 / 640.0, y1 / 480.0, x2 / 640.0, y2 / 480.0)

# Reads paths of images together with their labels
filename = 'training.csv'
image_list, label_list = read_input_file(filename)
label_list = np.array(label_list)
label_list = np.transpose(label_list)

def bb_intersection_over_union(y_true, y_pred):
    # determine the (x, y)-coordinates of the intersection rectangle
    true_x1 = K.slice(y_true, [0, 0], [-1, 1])
    true_y1 = K.slice(y_true, [0, 1], [-1, 1])
    true_x2 = K.slice(y_true, [0, 2], [-1, 1])
    true_y2 = K.slice(y_true, [0, 3], [-1, 1])

    pred_x1 = K.slice(y_pred, [0, 0], [-1, 1])
    pred_y1 = K.slice(y_pred, [0, 1], [-1, 1])
    pred_x2 = K.slice(y_pred, [0, 2], [-1, 1])
    pred_y2 = K.slice(y_pred, [0, 3], [-1, 1])

    true_x1 = K.reshape(true_x1, [-1])
    true_y1 = K.reshape(true_y1, [-1])
    true_x2 = K.reshape(true_x2, [-1])
    true_y2 = K.reshape(true_y2, [-1])

    pred_x1 = K.reshape(pred_x1, [-1])
    pred_y1 = K.reshape(pred_y1, [-1])
    pred_x2 = K.reshape(pred_x2, [-1])
    pred_y2 = K.reshape(pred_y2, [-1])

    xA = K.max((true_x1, pred_x1), axis=0)
    yA = K.max((true_y1, pred_y1), axis=0)
    xB = K.min((true_x2, pred_x2), axis=0)
    yB = K.min((true_y2, pred_y2), axis=0)

    # compute the area of intersection rectangle
    interArea = K.maximum((xB - xA + 1), 0) * K.maximum((yB - yA + 1), 0)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (true_x2 - true_x1 + 1) * (true_y2 - true_y1 + 1)
    boxBArea = (pred_x2 - pred_x1 + 1) * (pred_y2 - pred_y1 + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area

    union = boxAArea + boxBArea - interArea
    iou = interArea / (union + 0.00001)

    # return the intersection over union value
    return K.mean(iou)

train_X, valid_X, train_label, valid_label = train_test_split(image_list, label_list, test_size=0.2, random_state=13)

print(train_X.shape[0])
print(valid_X.shape)
print(train_label.shape)
print(valid_label.shape)

def generator(features, labels, batch_size):
    num_rows = features.shape[0]
    # Initialize a counter
    counter = 0
    cv_img = []
    while True:
        # for content, label in zip(features['content'], features['label']):
        for i in features:
            n = cv2.imread("images/" + i)
            #print(i)
            cv_img.append(n)
            
            #X_train[counter%batch_size] = n
            #y_train[counter%batch_size] = labels[counter]
            counter = counter + 1
            if(counter%batch_size == 0):
                cv_img = np.array(cv_img)
                cv_img = cv_img.astype('float32')
                train = cv_img / 255.
                cv_img = None
                cv_img = []
                yield train, labels[counter-batch_size:counter]


# In[2]:


train_generator = generator(train_X, train_label, batch_size)
val_generator = generator(valid_X, valid_label, batch_size)

train_samples=np.ceil(train_X.shape[0] / batch_size)
val_samples=np.ceil(valid_X.shape[0] / batch_size)

model = ResnetBuilder.build_resnet_9((3, 480, 640), 4)
model.compile(loss=keras.losses.mse, optimizer=keras.optimizers.Adam(), metrics=['mse', bb_intersection_over_union])
model.fit_generator(train_generator, steps_per_epoch=train_samples, epochs=1, verbose=1, 
                    validation_data=val_generator, validation_steps=val_samples)


# In[3]:


from keras.models import model_from_json


# In[4]:


model_json = model.to_json()
with open("model_1.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_1.h5")
print("Saved")


# In[5]:


json_file = open('model_1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("model_1.h5")
print("Loaded")
model.compile(loss=keras.losses.mse, optimizer=keras.optimizers.Adam(), metrics=['mse', bb_intersection_over_union])


# In[29]:


df1 = pd.read_csv('test.csv')

test = df1['image_name']


# In[ ]:


def generator(features, labels, batch_size):
    num_rows = features.shape[0]
    # Initialize a counter
    counter = 0
    cv_img = []
    while True:
        # for content, label in zip(features['content'], features['label']):
        for i in features:
            n = cv2.imread("images/" + i)
            #print(i)
            cv_img.append(n)
            
            #X_train[counter%batch_size] = n
            #y_train[counter%batch_size] = labels[counter]
            counter = counter + 1
            if(counter%batch_size == 0):
                cv_img = np.array(cv_img)
                cv_img = cv_img.astype('float32')
                train = cv_img / 255.
                cv_img = None
                cv_img = []
                yield train, labels[counter-batch_size:counter]


# In[41]:


def test_generator(features, batch_size):
    num_rows = features.shape[0]
    counter = 0
    cv_img = []
    # Initialize a counter
    while True:
        for i in features:
            n = cv2.imread("images/" + i)
            cv_img.append(n)
            counter = counter + 1
            if (counter%batch_size == 0):
                cv_img = np.array(cv_img)
                cv_img = cv_img.astype('float32')
                test1 = cv_img / 255.
                cv_img = None
                cv_img = []
                yield test1


# In[42]:


test_samples=np.ceil(test.shape[0] / batch_size)

pred = model.predict_generator(test_generator(test, batch_size), steps=test_samples, verbose=1)


# In[44]:


pred.shape


# In[45]:


test.shape


# In[48]:


pred[0]


# In[ ]:


[0.23278953 0.29601428 0.80433834 0.7575191 ]
[0.23278953 0.29601428 0.80433834 0.7575191 ]


# In[49]:


for i in range(len(pred)):
    print(pred[i])


# In[50]:


xf1 = np.floor(640*pred[:,0])
yf1 = np.floor(480*pred[:,1])
xf2 = np.floor(640*pred[:,2])
yf2 = np.floor(480*pred[:,3])


# In[51]:


final_labels = np.transpose(np.vstack((xf1,xf2,yf1,yf2)))


# In[57]:


final_labels_new = final_labels[:-1]
final_labels_new1 = final_labels[1:]


# In[61]:


final_labels_new.shape


# In[60]:


final_labels_new1.shape


# In[64]:


df1['x1'] = pd.DataFrame(final_labels_new1[:,0])
df1['x2'] = pd.DataFrame(final_labels_new1[:,1])
df1['y1'] = pd.DataFrame(final_labels_new1[:,2])
df1['y2'] = pd.DataFrame(final_labels_new1[:,3])

df1.to_csv('test2.csv',index=False)


# In[63]:


df1 = pd.read_csv('test.csv')


# In[ ]:





# In[ ]:


submission = pd.DataFrame({'fname': test.path.tolist(), 'label': labels})
submission.to_csv('submission.csv', index=False)


# In[ ]:





# In[ ]:




