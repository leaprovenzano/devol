import numpy as np
import random as rand
import math
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Flatten, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
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


class GenomeHandler:
    def __init__(self, max_conv_layers, max_dense_layers, max_filters, max_dense_nodes,
                 input_shape, n_classes, batch_normalization=True, dropout=True, max_pooling=True,
                 optimizers=None, activations=None, metric='accuracy'):
        if max_dense_layers < 1:
            raise ValueError(
                "At least one dense layer is required for softmax layer")
        filter_range_max = int(math.log(max_filters, 2)) + \
            1 if max_filters > 0 else 0
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
            list(range(3)) if max_pooling else 0,
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

        self.flatten_layers = [
            GlobalAveragePooling2D(), GlobalMaxPooling2D(), Flatten()]
        self.convolution_layers = max_conv_layers
        self.convolution_layer_size = len(self.convolutional_layer_shape)
        # this doesn't include the softmax layer, so -1
        self.dense_layers = max_dense_layers - 1
        self.dense_layer_size = len(self.dense_layer_shape)
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.metric = metric

    def mutate(self, genome, num_mutations):
        num_mutations = np.random.choice(num_mutations)
        for i in range(num_mutations):
            index = np.random.choice(list(range(1, len(genome))))
            if index < self.convolution_layer_size * self.convolution_layers:
                if genome[index - index % self.convolution_layer_size]:
                    range_index = index % self.convolution_layer_size
                    choice_range = self.convolutional_layer_shape[range_index]
                    genome[index] = np.random.choice(choice_range)
                elif rand.uniform(0, 1) <= 0.01:  # randomly flip deactivated layers
                    genome[index - index % self.convolution_layer_size] = 1
            elif index != len(genome) - 1:
                offset = self.convolution_layer_size * self.convolution_layers
                new_index = (index - offset)
                present_index = new_index - new_index % self.dense_layer_size
                if genome[present_index + offset]:
                    range_index = new_index % self.dense_layer_size
                    choice_range = self.dense_layer_shape[range_index]
                    genome[index] = np.random.choice(choice_range)
                elif rand.uniform(0, 1) <= 0.01:
                    genome[present_index + offset] = 1
            else:
                genome[index] = np.random.choice(
                    list(range(len(self.optimizer))))
        return genome

    def decode(self, genome):
        if not self.is_compatible_genome(genome):
            raise ValueError("Invalid genome for specified configs")
        model = Sequential()
        offset = 0
        input_layer = True
        for i in range(self.convolution_layers):
            if genome[offset]:
                convolution = None
                if input_layer:
                    convolution = Convolution2D(
                        genome[offset + 1], (3, 3),
                        padding='same',
                        input_shape=self.input_shape)
                    input_layer = False
                else:
                    convolution = Convolution2D(
                        genome[offset + 1], (3, 3),
                        padding='same')
                model.add(convolution)
                if genome[offset + 2]:
                    model.add(BatchNormalization())
                _activation = self.activation[genome[offset + 3]]
                model.add(Activation(_activation))
                model.add(Dropout(float(genome[offset + 4] / 20.0)))
                max_pooling_type = genome[offset + 5]
                if max_pooling_type == 1:
                    model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
            offset += self.convolution_layer_size

        if not input_layer:
            model.add(genome[offset])  # add flatten or global pool layer
            offset += 1

        for i in range(self.dense_layers):
            if genome[offset]:
                dense = None
                if input_layer:
                    dense = Dense(genome[offset + 1],
                                  input_shape=self.input_shape)
                    input_layer = False
                else:
                    dense = Dense(genome[offset + 1])
                model.add(dense)
                if genome[offset + 2]:
                    model.add(BatchNormalization())
                model.add(Activation(self.activation[genome[offset + 3]]))
                model.add(Dropout(float(genome[offset + 4] / 20.0)))
            offset += self.dense_layer_size

        model.add(Dense(self.n_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer=self.optimizer[genome[offset]],
                      metrics=["accuracy"])
        return model

    def generate(self):
        genome = []
        for i in range(self.convolution_layers):
            for r in self.convolutional_layer_shape:
                genome.append(np.random.choice(r))
        genome.append(np.random.choice(self.flatten_layers))
        for i in range(self.dense_layers):
            for r in self.dense_layer_shape:
                genome.append(np.random.choice(r))
        genome.append(np.random.choice(list(range(len(self.optimizer)))))
        genome[0] = 1
        return genome

    def is_compatible_genome(self, genome):
        expected_len = self.convolution_layers * self.convolution_layer_size \
            + self.dense_layers * self.dense_layer_size + 2
        if len(genome) != expected_len:
            return False
        ind = 0
        for i in range(self.convolution_layers):
            for j in range(self.convolution_layer_size):
                if genome[ind + j] not in self.convolutional_layer_shape[j]:
                    return False
            ind += self.convolution_layer_size
        ind+=1 #adding room for flatten  or global pooling layer
        for i in range(self.dense_layers):
            for j in range(self.dense_layer_size):
                if genome[ind + j] not in self.dense_layer_shape[j]:
                    return False
            ind += self.dense_layer_size
        if genome[ind] not in range(len(self.optimizer)):
            return False
        return True

    # metric = accuracy or loss
    def best_genome(self, csv_path, include_metrics=True):
        best = max if self.metric is "accuracy" else min
        col = -1 if self.metric is "accuracy" else -2
        data = np.genfromtxt(csv_path, delimiter=",")
        row = list(data[:, col]).index(best(data[:, col]))
        genome = list(map(int, data[row, :-2]))
        if include_metrics:
            genome += list(data[row, -2:])
        return genome

    # metric = accuracy or loss
    def decode_best(self, csv_path):
        return self.decode(self.best_genome(csv_path, metric, False))
