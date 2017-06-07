import numpy as np
import random as rand
import math
from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Dropout, Flatten, GlobalAveragePooling2D, GlobalMaxPooling2D, Input
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU, PReLU

from keras.optimizers import SGD
from tqdm import tqdm

##################################
# Genomes are represented as fixed-with lists of integers corresponding
# to sequential layers and properties. A model with 2 convolutional layers
# and 1 dense layer would look like:
#
# [<conv layer><conv layer><dense layer><optimizer>]
#
# The makeup of the convolutional layers and dense layers is defined in the
# GenomeHandler below under self.convolutional_layer_shape and
# self.dense_layer_shape. <optimizer> consists of just one property.
###################################


def _activation(activation):
    # a hacky check for advanced activations
    if type(activation) is not str:
        return activation()
    return Activation(activation)


def conv_layer(filters, kernal_size, activation, bn, strides=(1, 1)):
    def clayer(inp):
        x = Conv2D(filters, kernal_size, strides=strides, padding='same')(inp)
        if bn:
            x = BatchNormalization()(x)
        x = _activation(activation)(x)
        return x
    return clayer

def factorized_conv(filters, ksize, activation, bn, strides=(1, 1)):
    def block(inp):
        x = conv_layer(filters, (1, ksize), activation, bn, strides)(inp)
        x = conv_layer(filters, (ksize, 1), activation, bn, strides)(x)
        return x
    return block

def conv_block(n_layers, pooling, dropout, factorize, filters, ksize, activation, bn, strides=(1, 1)):
    if factorize:
        builder = factorized_conv
    else:
        builder = conv_layer

    def block(inp):
        x = inp
        for i in range(n_layers):
            x = builder(filters, ksize, activation, bn, strides)(x)
        x = Dropout(dropout)(x)

        if pooling == 1:
            x = MaxPooling2D(pool_size=(2, 2), padding="same")(x)
        elif pooling == 2:  # pool using convolution with stride of 2
            x = Conv2D(filters, (3, 3), strides=(
                2, 2), padding='same')(x)
        return x
    return block


class GenomeHandler:
    def __init__(self, max_conv_layers, max_dense_layers, max_filters, max_dense_nodes,
                 input_shape, n_classes, batch_normalization=True, dropout=True, max_pooling=True,
                 optimizers=None, activations=None):
        if max_dense_layers < 1:
            raise ValueError(
                "At least one dense layer is required for softmax layer")
        filter_range_max = int(math.log(max_filters, 2)) + int(max_filters > 0)
        self.optimizer = optimizers or [
            'adam',
            'rmsprop',
            'adagrad',
            'adadelta',
            'sgd',
            'nadam',
            SGD(lr=.001, decay=1e-6, momentum=0.9, nesterov=True)
        ]

        self.activation = activations or [
            'relu',
            'elu',
            'sigmoid',
            LeakyReLU,
            PReLU
        ]
        self.convolutional_layer_shape = [
            # Present
            [0, 1],
            # Filters
            [2**i for i in range(3, filter_range_max)],
            # Batch Normalization
            [0, (1 if batch_normalization else 0)],
            # Activation
            list(range(len(self.activation))),
            # Dropout
            [(i if dropout else 0) for i in range(11)],
            # Max Pooling
            list(range(3)) if max_pooling else [0],

            #factorize
            [0, 1],

            # kernal size
            [1, 3, 5],
            #layers per block
            [1, 2, 3]
        ]
        self.dense_layer_shape = [
            # Present
            [0, 1],
            # Number of Nodes
            [2**i for i in range(4, int(math.log(max_dense_nodes, 2)) + 1)],
            # Batch Normalization
            [0, (1 if batch_normalization else 0)],
            # Activation
            list(range(len(self.activation))),
            # Dropout
            [(i if dropout else 0) for i in range(11)],
        ]

        self.flatten_layers = [GlobalAveragePooling2D(), GlobalMaxPooling2D(), Flatten()]
        self.convolution_layers = max_conv_layers
        self.convolution_layer_size = len(self.convolutional_layer_shape)
        # this doesn't include the softmax layer, so -1
        self.dense_layers = max_dense_layers - 1
        self.dense_layer_size = len(self.dense_layer_shape)
        self.input_shape = input_shape
        self.n_classes = n_classes

    def mutate(self, genome, num_mutations):
        num_mutations = np.random.choice(num_mutations)
        for i in range(num_mutations):
            index = np.random.choice(list(range(1, len(genome))))

            if index < self.convolution_layer_size * self.convolution_layers:
                if genome[index - index % self.convolution_layer_size]:
                    range_index = index % self.convolution_layer_size
                    choice_range = self.convolutional_layer_shape[range_index]
                    # genome[index] = np.random.choice(choice_range)
                    genome[index] = rand.choice(choice_range)

                elif rand.uniform(0, 1) <= 0.01:  # randomly flip deactivated layers
                    genome[index - index % self.convolution_layer_size] = 1

            elif index == (self.convolution_layer_size * self.convolution_layers) :
                genome[index] = rand.choice(
                    [v for v in range(len(self.flatten_layers)) if v != genome[index]])

            elif index != len(genome) - 1:
                offset = (self.convolution_layer_size *
                          self.convolution_layers) + 1
                new_index = (index - offset)
                present_index = new_index - new_index % self.dense_layer_size
                if genome[present_index + offset]:
                    range_index = new_index % self.dense_layer_size
                    choice_range = self.dense_layer_shape[range_index]
                    genome[index] = rand.choice(choice_range)
                elif rand.uniform(0, 1) <= 0.01:
                    genome[present_index + offset] = 1
            else:
                genome[index] = rand.choice(
                    list(range(len(self.optimizer))))
        return genome



    def build_conv(self, encoding, kernal_size=(3, 3), scale_drop=.05):
        """build a basic convolutional layer according
        to parameters defined in encoding.
        """
        _, filters, bn, activation_index, drop, pool, factorize, ksize, n_layers = encoding
        activation = self.activation[activation_index]
        return conv_block(n_layers, pool, drop*scale_drop, 
                          factorize, filters, ksize, activation, bn)



    def build_dense(self, encoding, scale_drop=.08):
        """build a basic dense layer according
        to parameters defined in encoding.
        """
        _, nodes, bn, activation_index, drop = encoding
        activation = self.activation[activation_index]

        def block(inp):
            x = Dense(nodes)(inp)
            if bn:
                x = BatchNormalization()(x)
            x = _activation(activation)(x)
            x = Dropout(drop * scale_drop)(x)
            return x

        return block

    def decode(self, genome):
        if not self.is_compatible_genome(genome):
            raise ValueError("Invalid genome for specified configs")
        # initilize input layer
        input_layer = Input(shape=self.input_shape)
        x = input_layer  # set cursor to input layer so we don't have to bother checking

        # add conv layers
        for i in range(0, self.convolution_layers * self.convolution_layer_size, self.convolution_layer_size):
            params = genome[i:i + self.convolution_layer_size]
            if params[0]:
                x = self.build_conv(params)(x)

        # add out flatten or global pooling layer
        ix = (self.convolution_layers * self.convolution_layer_size)
        x = self.flatten_layers[genome[ix]](x)
        ix+=1
        # add dense layers
        for i in range(ix, len(genome), self.dense_layer_size):
            params = genome[i:i + self.dense_layer_size]
            if params[0]:
                x = self.build_dense(params)(x)
        output = Dense(self.n_classes, activation='softmax')(x)

        # initilize functional api model & compile
        model = Model(inputs=input_layer, outputs=output)
        model.compile(loss='categorical_crossentropy',
                      optimizer=self.optimizer[genome[-1]],
                      metrics=["accuracy"])
        return model

    def generate(self):
        genome = []
        for i in range(self.convolution_layers):
            for r in self.convolutional_layer_shape:
                # genome.append(np.random.choice(r))
                genome.append(rand.choice(r))

        genome.append(rand.randint(0, len(self.flatten_layers)-1))
        for i in range(self.dense_layers):
            for r in self.dense_layer_shape:
                genome.append(rand.choice(r))
        genome.append(rand.choice(range(len(self.optimizer))))
        genome[0]=1
        return genome


    def is_compatible_genome(self, genome):
        expected_len=self.convolution_layers * self.convolution_layer_size + self.dense_layers * self.dense_layer_size + 2
        if len(genome) != expected_len:
            return False
        ind=0
        for i in range(self.convolution_layers):
            for j in range(self.convolution_layer_size):
                if genome[ind + j] not in self.convolutional_layer_shape[j]:
                    return False
            ind += self.convolution_layer_size
        ind += 1  # adding room for flatten  or global pooling layer
        if genome[ind] not in range(len(self.flatten_layers)):
            return False
        for i in range(self.dense_layers):
            for j in range(self.dense_layer_size):
                if genome[ind + j] not in self.dense_layer_shape[j]:
                    return False
            ind += self.dense_layer_size

        if genome[ind] not in range(len(self.optimizer)):
            return False
        return True

    # metric = accuracy or loss
    def best_genome(self, csv_path, metric='accuracy', include_metrics=True):
        best=max if metric is "accuracy" else min
        col=-1 if metric is "accuracy" else -2
        data=np.genfromtxt(csv_path, delimiter=",")
        row=list(data[:, col]).index(best(data[:, col]))
        genome=list(map(int, data[row, :-2]))
        if include_metrics:
            genome += list(data[row, -2:])
        return genome

    # metric = accuracy or loss
    def decode_best(self, csv_path):
        return self.decode(self.best_genome(csv_path, metric, False))
