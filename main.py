import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow import keras
import keras
from keras import utils

class DenseNN(tf.Module):
    def __init__(self, outputs, activate="relu"):
        super().__init__()
        self.outputs = outputs
        self.activate = activate
        self.fl_init = False

    def __call__(self, x):
        if not self.fl_init:
            self.w = tf.Variable(
                tf.random.truncated_normal((x.shape[-1], self.outputs), stddev = 0.1), name = "w")
            self.b = tf.Variable(tf.zeros([self.outputs], dtype = tf.float32), name = "b")
            self.fl_init = True

        y = x @ self.w + self.b
    
        if self.activate == "relu":
            return tf.nn.relu(y)
        elif self.activate == "softmax":
            return tf.nn.softmax(y)
        return y

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = tf.cast(x_train, tf.float32) / 255.0
x_test = tf.cast(x_test, tf.float32) / 255.0
x_train = tf.reshape(x_train, [-1, 28*28])
x_test = tf.reshape(x_test, [-1, 28*28])
y_train = utils.to_categorical(y_train, 10)

layer_1 = DenseNN(128)
layer_2 = DenseNN(10, activate="softmax")

def model_predict(x):
    y = layer_1(x)
    return layer_2(y)

cross_entropy = lambda y_true, y_pred: tf.reduce_mean(
    tf.keras.losses.categorical_crossentropy(y_true, y_pred))
opt = tf.optimizers.Adam(learning_rate=0.001)

BATCH_SIZE = 32
EPOCHS = 10
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE)

# Обучение модели
for epoch in range(EPOCHS):
    total_loss = 0
    for x_batch, y_batch in train_dataset:
        with tf.GradientTape() as tape:
            y_pred = model_predict(x_batch)
            loss = cross_entropy(y_batch, y_pred)
        
        grads = tape.gradient(loss, layer_1.trainable_variables + layer_2.trainable_variables)
        opt.apply_gradients(zip(grads, layer_1.trainable_variables + layer_2.trainable_variables))
        total_loss += loss
    
    print(f"Epoch {epoch + 1}, Loss: {total_loss.numpy() / len(train_dataset)}")

y_pred = model_predict(x_test)
y_pred_labels = tf.argmax(y_pred, axis=1).numpy()
accuracy = tf.reduce_mean(tf.cast(tf.equal(y_pred_labels, y_test), tf.float32)).numpy() * 100
print(f"Test Accuracy: {accuracy:.2f}%")