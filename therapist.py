import tensorflow as tf
import pandas as pd
from tensorflow import keras
import numpy as np
import io
# How I originally created the model:
from google.colab import files
# data taken from https://www.kaggle.com/datasets/nelgiriyewithana/emotions
uploaded = files.upload()
df = pd.read_csv(io.BytesIO(uploaded['emotiondata.csv']))
print(df)

train_dataset = df.sample(frac = 0.5)
test_dataset = df.drop(train_dataset.index)

def preprocess(df):
    data = df.copy()
    data.pop('Unnamed: 0')
    data.loc[data['label'] == '0','label'] = 0
    data.loc[data['label'] == '1','label'] = 1
    data.loc[data['label'] == '2','label'] = 2
    data.loc[data['label'] == '3','label'] = 3
    data.loc[data['label'] == '4','label'] = 4
    data.loc[data['label'] == '5','label'] = 5
    labels = data.pop('label').astype(np.float32)
    data = data.values
    return data, labels

train_data, train_labels = preprocess(train_dataset)
valid_data, valid_labels = preprocess(test_dataset)

encoder = tf.keras.layers.TextVectorization()
encoder.adapt(train_data)

model = tf.keras.Sequential([
    encoder,
    tf.keras.layers.Embedding(
        input_dim=len(encoder.get_vocabulary()),
        output_dim=64,
        # Use masking to handle the variable sequence lengths
        mask_zero=True),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, dropout=0.1)),
    tf.keras.layers.Dense(6, activation='softmax')
])

model.summary()

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode="min", patience=3, restore_best_weights=True)
history = model.fit(x=train_data,y=train_labels, epochs=8, validation_data=(valid_data,valid_labels), callbacks=[callback])

model.save()
