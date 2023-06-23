import pandas as pd
import cv2
import numpy as np


def test_generator(features, batch_size):
    num_rows = features.shape[0]
    cv_img = []
    # Initialize a counter
    while True:
        counter = 0
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
            elif counter==num_rows:
                cv_img = np.array(cv_img)
                cv_img = cv_img.astype('float32')
                test1 = cv_img / 255.
                yield test1




# def test_images(filename):
#     test_image = []
#     c = 0
#     df1 = pd.read_csv(filename)
#     for i in df1['image_name'].values:
#         n = cv2.imread("images/" + i)
#         c = c + 1
#         test_image.append(n)
#         # print(c)
#         if c == 100:
#             break

#     test_image = np.array(test_image)
#     test_image = test_image.astype('float32')
#     test_image = test_image / 255.

#     return test_image


def write_to_file(input_file_name, output_file_name, pred):
    
    x1 = np.floor(640 * pred[:, 0])
    y1 = np.floor(480 * pred[:, 1])
    x2 = np.floor(640 * pred[:, 2])
    y2 = np.floor(480 * pred[:, 3])

    final_labels = np.transpose(np.vstack((x1, x2, y1, y2)))
    print (final_labels.shape)

    df1 = pd.read_csv(input_file_name)
    df1['x1'] = pd.DataFrame(final_labels[:,0])
    df1['x2'] = pd.DataFrame(final_labels[:,1])
    df1['y1'] = pd.DataFrame(final_labels[:,2])
    df1['y2'] = pd.DataFrame(final_labels[:,3])
    df1.to_csv(output_file_name, index=False)
