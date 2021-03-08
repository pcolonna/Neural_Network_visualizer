import numpy as np
import tensorflow as tf
import config

def create_model(num_layers, hidden_units_per_layers):
    model = tf.keras.models.Sequential()
    
    model.add(tf.keras.layers.Dense(hidden_units_per_layers, activation='sigmoid', input_shape=(784,)))
    
    for i in range(num_layers - 1):
        model.add(tf.keras.layers.Dense(hidden_units_per_layers, activation='sigmoid'))
    
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def train(num_layers, hidden_units_per_layers, epochs, batch_size):

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = np.reshape(x_train, (60000, 784))
    x_train = x_train / 255.

    x_test = np.reshape(x_test, (10000, 784))
    x_test = x_test / 255.

    model = create_model(num_layers, hidden_units_per_layers)
    print(model.summary())

    _ = model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=epochs, batch_size=batch_size,
        verbose=2
    )

    model.save(config.model_path)

    return model