import os

import flask
import numpy as np

from flask import request, jsonify
from flask_cors import CORS, cross_origin


def softmax(x):
    """
    Applies the Softmax activation function to 'x'.

    Args:
        x : Inputs of shape (batch_size, n_classes).

    Returns:
        z : Inputs after the Softmax function is applied. Shape of (batch_size, n_classes).
    """

    return np.exp(x) / np.sum(np.exp(x), axis=1).reshape((-1, 1))


class ReLU:
    def forward(self, z):
        """
        Args:
            z : Output from the previous layer. Shape of (batch_size, features).

        Returns:
            y : Input after the activation function. Shape of (batch_size, features).
        """

        # Note: Keep for gradients
        self.z = z

        return np.maximum(z, np.zeros(z.shape))

    def backward(self, dy):
        """
        Args:
            dy : Gradient of the loss w.r.t. 'y'. Shape of (batch_size, features).

        Returns:
            dz : Gradient of the loss w.r.t. 'z'. Shape of (batch_size, features).
        """

        return np.where(self.z > 0, np.ones(self.z.shape), np.zeros(self.z.shape)) * dy


class Linear:
    def __init__(self, in_features, out_features):
        """
        Args:
            in_features  : Number of input features.
            out_features : Number of output features.
        """

        bound = np.sqrt(6 / (in_features + out_features))

        # Initialise the weights and biases
        self.W = np.random.uniform(-bound, bound, (out_features, in_features))
        self.b = np.random.uniform(-bound, bound, (out_features))

        # Internal gradients for learning
        self.dW = None
        self.db = None

    def forward(self, x, W=None, b=None):
        """
        Args:
            x : Inputs of shape (batch_size, in_features).
            W : Optional weights of shape (out_features, in_features).
            b : Optional bias terms of shape (out_features).

        Returns:
            z : Outputs of shape (batch_size, out_features).
        """

        # Decide which weights and biases to use
        _W = self.W if W is None else W
        _b = self.b if b is None else b

        # Keep for gradient
        self.x = x

        return x @ _W.T + _b.T

    def backward(self, dy):
        """
        Args:
            dy : Gradient of the loss w.r.t. 'y'. Shape of (batch_size, out_features).

        Returns:
            dx : Gradient of the loss w.r.t. 'x'. Shape of (batch_size, in_features).
        """

        # Compute the internal gradients
        self.dW = dy.T @ self.x
        self.db = np.sum(dy.T, axis=1)

        return dy @ self.W

    def save(self, id, folder):
        """
        Saves the linear layer's configuration to disk.

        Args:
            id     : An unique ID used for saving the model to disk.
            folder : Base path for the folder to save to.
        """

        np.save('{}/{}_W'.format(folder, id), self.W)
        np.save('{}/{}_b'.format(folder, id), self.b)

    def load(self, id, folder):
        """
        Loads the linear layer's configuration from disk.

        Args:
            id     : An unique ID used for loading the model from disk.
            folder : Base path for the folder to load from.
        """

        self.W = np.load('{}/{}_W.npy'.format(folder, id))
        self.b = np.load('{}/{}_b.npy'.format(folder, id))


class MLP:
    def __init__(self, name, in_features, hidden_sizes, out_features, activation_fn):
        """
        Args:
            name          : Name of the network.
            in_features   : Number of input features.
            hidden_sizes  : An array of hidden layer sizes.
            out_features  : Number of output features.
            activation_fn : Activation function to use for each layer.
        """

        self.name = name

        # Initialise the first hidden layer
        self.modules = [
            Linear(in_features, hidden_sizes[0]),
            activation_fn()
        ]

        # Initialise the rest of the hidden layers
        for i in range(len(hidden_sizes) - 1):
            self.modules.append(Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            self.modules.append(activation_fn())

        # Initialise the output layer
        self.modules.append(Linear(hidden_sizes[-1], out_features))

    def forward(self, x):
        """
        Do the forward pass.

        Args:
            x : Inputs of shape (batch_size, in_features).

        Returns:
            y : Outputs of shape (batch_size, out_features).
        """

        y = x

        for layer in self.modules:
            y = layer.forward(y)

        return y

    def backward(self, dy):
        """
        Do the backward pass.

        Args:
            dy : Gradient of the loss w.r.t. 'y'. Shape of (batch_size, out_features).

        Returns:
            dx : Gradient of the loss w.r.t. 'x'. Shape of (batch_size, in_features).
        """

        dx = dy

        for i in range(len(self.modules) - 1, -1, -1):
            dx = self.modules[i].backward(dx)

        return dx

    def save(self, folder=None):
        """
        Save the network to disk.
        """

        for i, module in enumerate(self.modules):
            if hasattr(module, 'W'):
                module.save(self.get_id(
                    i), folder if folder is not None else self.get_folder())

    def load(self, folder=None):
        """
        Load the network from disk.
        """

        for i, module in enumerate(self.modules):
            if hasattr(module, 'W'):
                module.load(self.get_id(
                    i), folder if folder is not None else self.get_folder())

    def get_folder(self):
        return './model'

    def get_id(self, layer_index):
        return '{}_layer-{}'.format(self.name, layer_index)


def main():
    # Load the network to memory
    net = MLP('mnist', 784, [512, 256], 10, activation_fn=ReLU)
    net.load(folder='../model')

    # Instantiate the webserver
    app = flask.Flask(__name__)

    # Setup CORS
    CORS(app)

    # Define a route for predictions
    @app.route('/', methods=['POST'])
    @cross_origin()
    def predict_route():
        # Read the request data into a Numpy array
        pixels = request.json

        x = np.array(pixels)
        x = x.flatten().reshape((1, -1))

        # Pass the data through the network and compute the probabilities for each class
        probabilities = softmax(net.forward(x))

        return jsonify(probabilities[0, :].tolist())

    app.run(host='0.0.0.0', port=os.getenv('PORT', 8080))


main()
