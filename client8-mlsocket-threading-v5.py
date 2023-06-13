from mlsocket import MLSocket
import socket
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os


HOST = 'localhost'
PORT = 9999


client_n = int(os.path.basename(__file__)[6])
print(client_n)

#importing and preprocessing data
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
assert X_train.shape == (60000, 28, 28)
assert X_test.shape == (10000, 28, 28)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,)

X_train = X_train/255
X_train = X_train.astype(float)
y_train = y_train.astype(int)

# Assign each client with training images from two classes
if client_n == 0:
    idx0 = y_train == client_n
    idx1 = y_train == 9
else:
    idx0 = y_train == client_n
    idx1 = y_train == client_n - 1

X_train_client = np.concatenate((X_train[idx0][:len(X_train[idx0])//2], X_train[idx1][len(X_train[idx1])//2:]))
y_train_client = np.concatenate((y_train[idx0][:len(y_train[idx0])//2], y_train[idx1][len(y_train[idx1])//2:]))

X_train_client_cnn = np.expand_dims(X_train_client, -1)

print('Shape of client {} data: '.format(client_n), X_train_client.shape)
print("Client labels", y_train_client)

tf.random.set_seed(1)
keras.backend.clear_session()

# Define initialise CNN model function
def initialiseCNN(lr=0.1):
    model = keras.Sequential([

        keras.Input(shape=(28, 28, 1)),

        keras.layers.Conv2D(32, kernel_size=(5, 5), padding='same', activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'),

        keras.layers.Conv2D(64, kernel_size=(5, 5), padding='same', activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'),

        keras.layers.Flatten(),
        keras.layers.Dense(units=512, activation='relu'),
        keras.layers.Dense(10, activation="softmax"),
    ])

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=keras.optimizers.SGD(learning_rate=lr),
                  metrics=['accuracy'])

    return model

def trainClient(client_model, batch_size = 10, epochs = 5):

    print("Training client ", str(client_n))
    client_model.fit(X_train_client_cnn, y_train_client, shuffle = True,
                                     batch_size=batch_size,epochs=epochs)

client_model = initialiseCNN(lr=0.125)     # change learning rate to desired value

client_ip = '127.0.0.1'   # set to local host for simulation on local machine
client_port = 9988        # each client is assigned a port for differentiation on server side
communication_round =100

for _ in range(communication_round):
    print("labels for training: ", y_train_client)
    trainClient(client_model)

    # send model to server
    with MLSocket() as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((client_ip, client_port))
        s.connect((HOST, PORT))
        s.send(client_model)
        s.send(str(X_train_client_cnn.shape[0]).encode())
        print("Model sent to server.")

    # reverse connection to receive global model from server
    with MLSocket() as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((client_ip, client_port))
        s.listen()
        print("Client socket is listening")
        conn, addr = s.accept()
        updatedModel = conn.recv(1024)
        print('Global model received')

    client_model.set_weights(updatedModel.get_weights())
