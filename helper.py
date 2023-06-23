import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from keras import backend as K
import random

def read_input_file(filename):
    df = pd.read_csv(filename)
    l = df['image_name'].values
    x1, x2, y1, y2 = df['x1'].values, df['x2'].values, df['y1'].values, df['y2'].values
    return l, (x1 / 640.0, y1 / 480.0, x2 / 640.0, y2 / 480.0)


# def get_train_data(image_list, label_list):
#     cv_img = []
#     c = 0
#     for i in image_list:
#         n = cv2.imread("images/" + i)
#         c = c + 1
#         cv_img.append(n)
#         if c == 50:
#             break

#     cv_img = np.array(cv_img)
#     label_list = np.array(label_list)
#     label_list = np.transpose(label_list)

#     # print(label_list.shape)
#     # print(cv_img.shape)

#     return cv_img, label_list, c



# def split_float_data(train_data, label_list, c):

#     train_X, valid_X, train_label, valid_label = train_test_split(train_data, label_list[:c], test_size=0.2, random_state=13)
#     train_X = train_X.astype('float32')
#     valid_X = valid_X.astype('float32')
#     train_X = train_X / 255.
#     valid_X = valid_X / 255.

#     del train_data
#     del label_list

#     return train_X, valid_X, train_label, valid_label

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

def generator(features, labels, batch_size):
    num_rows = features.shape[0]
    cv_img = []
    while True:
        counter = 0
        x = list(zip(features,labels))
        random.shuffle(x)
        features,labels=zip(*x)
        features = np.array(features)
        labels = np.array(labels)
        # for content, label in zip(features['content'], features['label']):
        for i in features:
            n = cv2.imread("images/" + i)
            cv_img.append(n)
            counter = counter + 1
            if(counter%batch_size == 0):
                cv_img = np.array(cv_img)
                cv_img = cv_img.astype('float32')
                train = cv_img / 255.
                cv_img = None
                cv_img = []
                yield train, labels[counter-batch_size:counter]
            elif counter==num_rows:
                cv_img = np.array(cv_img)
                cv_img = cv_img.astype('float32')
                train = cv_img / 255.
                cv_img = None
                cv_img = []
                yield train, labels[counter-counter%batch_size:counter]

