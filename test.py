from keras.models import model_from_json
from testing import *
from helper import *
import keras
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

batch_size = 8
pick = 4


# Load Model
json_file = open('model_'+str(pick)+'.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("model_"+str(pick)+".h5")
print("Loaded model from disk")

# Compile
model.compile(loss=keras.losses.mse, optimizer=keras.optimizers.Adam(), metrics=['mse', bb_intersection_over_union])

input_file_name = 'test.csv'
df1 = pd.read_csv(input_file_name)
test = df1['image_name']
test_samples=np.ceil(test.shape[0] / batch_size)

# Predict Bounding Box
pred = model.predict_generator(test_generator(test, batch_size), steps=test_samples, verbose=1)

# Output file
output_file_name = str(pick)+'.csv'
write_to_file(input_file_name, output_file_name, pred)
