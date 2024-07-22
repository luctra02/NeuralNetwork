import pickle
from typing import Dict, List, Any, Union
import numpy as np
# Keras
from tensorflow import keras
from keras.utils import pad_sequences



def load_data() -> Dict[str, Union[List[Any], int]]:
    path = "keras-data.pickle"
    with open(file=path, mode="rb") as file:
        data = pickle.load(file)

    return data


def preprocess_data(data: Dict[str, Union[List[Any], int]]) -> Dict[str, Union[List[Any], np.ndarray, int]]:
    """
    Preprocesses the data dictionary. Both the training-data and the test-data must be padded
    to the same length; play around with the maxlen parameter to trade off speed and accuracy.
    """

    #increased the maxlen by dividing 12 instead of 16
    maxlen = data["max_length"]//12
    data["x_train"] = pad_sequences(data['x_train'], maxlen=maxlen)
    data["y_train"] = np.asarray(data['y_train'])
    data["x_test"] = pad_sequences(data['x_test'], maxlen=maxlen)
    data["y_test"] = np.asarray(data['y_test'])

    return data


def train_model(data: Dict[str, Union[List[Any], np.ndarray, int]], model_type="feedforward") -> float:
    """
    Build a neural network of type model_type and train the model on the data.
    Evaluate the accuracy of the model on test data.

    :param data: The dataset dictionary to train neural network on
    :param model_type: The model to be trained, either "feedforward" for feedforward network
                        or "recurrent" for recurrent network
    :return: The accuracy of the model on test data
    """

    # TODO build the model given model_type, train it on (data["x_train"], data["y_train"])
    #  and evaluate its accuracy on (data["x_test"], data["y_test"]). Return the accuracy

    #Define the model
    model = keras.Sequential()

    #Embed the word-vectors into a high-dim metric space
    model.add(keras.layers.Embedding(input_dim=data['vocab_size'], output_dim=128))

    if model_type == "feedforward":
        # Flatten layer for feedforward network
        model.add(keras.layers.Flatten(name="Flatten"))
        # Dense layers

        #128 nodes in the first hidden layer with relu as function
        model.add(keras.layers.Dense(128, activation='relu'))
        #1 node for output with sigmoid as function, which tells us if the review is positive(activated) or not (not activated) based on the previous activations and the sigmoid function
        model.add(keras.layers.Dense(1, activation='sigmoid'))
    
    elif model_type == "recurrent":
        # LSTM layer for recurrent network with 64 nodes
        model.add(keras.layers.LSTM(64))
        #Dense layer for output with one node
        model.add(keras.layers.Dense(1, activation='sigmoid'))

    model.compile(
        loss="binary_crossentropy",  # Use binary crossentropy for binary classification
        optimizer=keras.optimizers.Adam(
            learning_rate=0.001
        ),  # Optimizer to use.
        metrics=[
            "accuracy"
        ],  # Metrics to survey during training. Here we are interested in accuracy
    )

    model.fit(data['x_train'], data['y_train'], epochs=2, batch_size=128)

    test_loss, test_accuracy = model.evaluate(data['x_test'], data['y_test'])

    return test_accuracy






def main() -> None:
    print("1. Loading data...")
    keras_data = load_data()
    print("2. Preprocessing data...")
    keras_data = preprocess_data(keras_data)
    print("3. Training feedforward neural network...")
    fnn_test_accuracy = train_model(keras_data, model_type="feedforward")
    print('Model: Feedforward NN.\n'
          f'Test accuracy: {fnn_test_accuracy:.3f}')
    print("4. Training recurrent neural network...")
    rnn_test_accuracy = train_model(keras_data, model_type="recurrent")
    print('Model: Recurrent NN.\n'
          f'Test accuracy: {rnn_test_accuracy:.3f}')



if __name__ == '__main__':
    main()

