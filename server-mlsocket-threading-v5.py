import socket
import threading
import numpy as np
from mlsocket import MLSocket
import time
from tensorflow import keras
from sklearn.metrics import accuracy_score

# Load data and test set
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
assert X_train.shape == (60000, 28, 28)
assert X_test.shape == (10000, 28, 28)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,)
X_test = X_test/255
X_test = X_test.astype(float)
y_test = y_test.astype(int)

# Define function to initialise global model
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
# Define the aggregation function
def aggregateModel(models_dataSize, client_importance):
    models = [x[0] for x in models_dataSize]
    client_model_weights = [np.array(model.get_weights(), dtype=object) for model in models]
    client_model_weighted_weights = [x * y for x, y in zip(client_model_weights, client_importance)]
    global_weights = sum(client_model_weighted_weights)
    return global_weights

# Define the client importance calculation function
def calculateClientImportance(models_dataSize):
    client_data_size = [x[1] for x in models_dataSize]
    client_importance = [x/sum(client_data_size) for x in client_data_size]
    return client_importance

# Define the function to evaluate global model accuracy
def modelAccuracy(model):
    prob = model.predict(X_test)
    pred = np.argmax(prob, axis=-1)

    print("Model can predict classes: ", np.unique(pred))
    print("Model accuracy: {}".format(accuracy_score(y_test, pred)),
          "\n-------------------------------------------------------------")
    return accuracy_score(y_test, pred)

# Define the function to handle each client's connection in the receive model stage
def handleClientConnection(conn, addr, models_dataSize):
    print(f"[Receiving] Connection with {addr} connected.")

    model = conn.recv(1024) # Receive the model information from the client
    client_datasize = conn.recv(1024) # Receive the number of training samples from the client
    models_dataSize.append((model, int(client_datasize.decode())))

    conn.close()
    print(f"[Receiving] Connection with {addr} disconnected.")

# Define the function to send the global model back to clients
def sendGlobalModel(IP):
    with MLSocket() as s:
        s.connect(IP)
        s.send(global_model)
        s.close()
    print(f"[Sending] Global model sent to {IP}.")


# Initialize the server
server_socket = MLSocket()
server_socket.bind(('localhost', 9999))
server_socket.listen()
print("[STARTING] Server is starting...")

# Initialize the models and client datasize
global_model = initialiseCNN()

communication_round = 100     # change to desired value
number_of_clients = 2      # change to desired value
global_accuracy = []

for round in range(communication_round):
    print(f"Communication round {round+1}:")

    # Initialize a list to keep track of each client's connection
    client_IPs = []
    models_dataSize = []

    while len(client_IPs) != number_of_clients:
        # Wait for a new client to connect
        conn, addr = server_socket.accept()

        # Block connection from client who has already send model
        if addr in client_IPs:
            conn.close()
            print(f'Connection from client {addr} rejected.')
        # Add the client's connection to the list
        else:
            client_IPs.append(addr)

            # Create a new thread to handle the client's connection
            client_thread = threading.Thread(target=handleClientConnection, args=(conn, addr, models_dataSize))
            client_thread.start()

    while len(models_dataSize) != number_of_clients:
        time.sleep(1)
    # Aggregate models
    client_importance = calculateClientImportance(models_dataSize)
    print("Client importance: ", client_importance)
    aggregated_weights = aggregateModel(models_dataSize, client_importance)
    global_model.set_weights(aggregated_weights)  # Set the new weights for the server's model for calculating accuracy

    # Send global model back to each client
    for IP in client_IPs:
        client_thread = threading.Thread(target = sendGlobalModel, args = (IP,))
        client_thread.start()

    print(f'Training result at round {round+1}:')
    model_accuracy = modelAccuracy(global_model)
    global_accuracy.append(model_accuracy)
    print("Model accuracy history: ", global_accuracy)

import csv

data = [[x] for x in global_accuracy]
file = open('accuracy.csv', 'w+', newline='')

with file:
    write = csv.writer(file)
    write.writerows(data)
