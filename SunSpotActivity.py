import csv
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image  as mpimg

import numpy as np


tf.keras.backend.clear_session()  #clear any internal variable
tf.random.set_seed(51)
np.random.seed(51)

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


def plot_series(time, series, color, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format, color=color)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)

def window_data(series, window_size, batch_size, shuffle_buffer):
    '''
    Returns data given as series in windows of window_size dim.

    Parameters:
        series (np.array) :series data
        window_size (int) : size of window
        batch_size (int)
        shuffle_buffer (int)
    Returns:
        series in batches
    '''
    series = tf.expand_dims(series, axis=-1)
    data = tf.data.Dataset.from_tensor_slices(series)
    data = data.window(size=window_size+1, shift=1, drop_remainder=True)
    data = data.flat_map(lambda w: w.batch(window_size+1))
    data = data.shuffle(shuffle_buffer)
    data = data.map(lambda w: (w[:-1], w[1:]))
    return data.batch(batch_size).prefetch(1)

def model_forecast(model, series, window_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(32).prefetch(1)
    forecast = model.predict(ds)
    return forecast

data_path = '/home/kiagkons/tensorflow_preparation/Sunspot/sunspots.csv'

index = []
activity = []
# Load data and split into index values and sun spot activity
with open(data_path,'r') as csv_file:
    reader = csv.reader(csv_file, delimiter = ',')
    next(reader)
    for row in reader:
        index.append(int(row[0]))
        activity.append(float(row[2]))

# convert list to numpy arrays as it is easier to work with
index = np.array(index)
activity = np.array(activity)

# Plot data to inspect for seasonality, trend, noise
# plt.figure(figsize=(10, 6))
# plt.plot(index,activity)
# plt.show()


# Number of training samples 3235
print(len(index))

samples = 3000
window_size = 80
shuffle_buffer = 1000
batch_size = 120

x_train = activity[:samples]
time_train = index[:samples]
x_valid = activity[samples:]
time_valid = index[samples:]

train_data = window_data(x_train,window_size=window_size,
                         batch_size=batch_size,
                         shuffle_buffer=shuffle_buffer)

model=tf.keras.models.Sequential([

    tf.keras.layers.Conv1D(filters=60, kernel_size=5,
                           padding='causal',
                           activation='relu',
                           input_shape=[None,1]),
    tf.keras.layers.LSTM(60,return_sequences=True),
    tf.keras.layers.LSTM(60, return_sequences = True),
    tf.keras.layers.Dense(20,activation='relu'),
    tf.keras.layers.Dense(10,activation='relu'),
    tf.keras.layers.Dense(1),
    tf.keras.layers.Lambda(lambda x: x * 400)
])

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * 10**(epoch / 20))

optimizer = tf.keras.optimizers.SGD(lr=1e-5, momentum=0.9)
model.compile(loss = tf.keras.losses.Huber(),
              optimizer=optimizer,
              metrics=['mae']
              )

history = model.fit(train_data,
                    epochs = 500)
                    # callbacks=[lr_scheduler])

# plot learning rate vs loss to get proper loss for training
# plt.semilogx(history.history["lr"], history.history["loss"])
# plt.axis([1e-8, 1e-2, 0, 80])
# plt.show()


rnn_forecast = model_forecast(model, activity[..., np.newaxis], window_size)
rnn_forecast = rnn_forecast[samples - window_size:-1, -1, 0]

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid, color='b')
plot_series(time_valid, rnn_forecast, color='r')
plt.show()

print('Mean average error is :',tf.keras.metrics.mean_absolute_error(x_valid, rnn_forecast).numpy())

print(rnn_forecast)
