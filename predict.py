import tensorflow as tf
from tensorflow import keras
import numpy as np


print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))

model = keras.models.load_model('my_model_relu')

doggy_path = "img_3.png"
class_names = ['кошка', 'собака']
img_height = 180
img_width = 180


img = tf.keras.utils.load_img(
    doggy_path, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "На изображении {} с вероятностью {:.2f} процентов."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
