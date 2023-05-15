import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

import numpy as np
import pandas as pd

data = pd.read_excel ('DummyData.xlsx')
labels = np.array (data.pop ('Dark Count'))
data = np.array (data)
training_size = (int)(data.size * 0.8)

training_data = data [0:training_size]
training_labels = labels [0:training_size]
testing_data = data [training_size:]
testing_labels = labels [training_size:]

normalizer = preprocessing.Normalization()

normalizer.adapt(np.array(training_data))

model = tf.keras.Sequential ([
    normalizer,
    layers.Dense (32, activation='relu'),
    layers.Dense (32, activation='relu'),
    layers.Dense (1)
])

loss = keras.losses.MeanAbsoluteError ()
metrics = ['accuracy']
model.compile (loss = loss, optimizer = tf.keras.optimizers.Adam (0.001), metrics = metrics)
print (model.summary ())

val_dataSize = (int)(training_data.size * 0.2)
val_data = training_data []
training_data = training_data []

model.fit (training_data, training_labels, epochs = 100, verbose = 2, validation_data = (testing_data, testing_labels))
