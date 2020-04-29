import tensorflow as tf
from tensorflow.keras.models import load_model
import torch
import torch.nn as nn
import numpy as np

# Project files
from network import Net


def main():
    keras_model = load_model(
        "model.h5",
        custom_objects={
            "tf": tf
        })
    pytorch_model = Net()

    number_of_keras_parameters = keras_model.count_params()
    number_of_pytorch_parameters = sum(
        p.numel() for p in pytorch_model.parameters() if p.requires_grad)

    if number_of_keras_parameters != number_of_pytorch_parameters:
        print("\n\nNot the same number of trainable parameters in the models")
        print("Keras:   {}\nPytorch: {}".format(
            number_of_keras_parameters, number_of_pytorch_parameters))
        return

    print("Keras layer names:")
    weights = []
    biases = []
    for layer in keras_model.layers:
        if len(layer.trainable_weights) > 0:
            print(layer.name)
            layer_weights_and_biases = layer.get_weights()
            weights.append(layer_weights_and_biases[0])
            if len(layer_weights_and_biases) > 1:
                biases.append(layer_weights_and_biases[1])

    print("\n\nPytorch layer names:")
    for name, parameters in pytorch_model.named_parameters():
        print(name)
        if name.split('.')[-1] == "weight":
            parameters.data = torch.from_numpy(np.moveaxis(np.moveaxis(
                weights.pop(0), 2, 0), 3, 0))
        elif name.split('.')[-1] == "bias":
            parameters.data = torch.from_numpy(biases.pop(0))
    torch.save(pytorch_model, "pytorch_model.pt")


main()
