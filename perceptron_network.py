import numpy as np
from create_selection import *
from math import atan
from random import random
from random import seed
from time import time
import asyncio


class Selection:
    @property
    def two_args(self):
        return [[0, 0], [0, 1], [1, 0], [1, 1]]

    @property
    def four_args(self):
        shit = []
        for i1 in range(2):
            for i2 in range(2):
                for i3 in range(2):
                    for i4 in range(2):
                        shit.append([i1, i2, i3, i4])
        return shit

    @property
    def bin_four(self):
        return [[i / 15] for i in range(16)]

    @property
    def bin_four_seq(self):
        return [[0 if j != i else 1 for j in range(16)] for i in range(16)]

    @property
    def xor_(self):
        return [[0], [1], [1], [0]]

    @property
    def and_(self):
        return [[0], [0], [0], [1]]

    @property
    def or_(self):
        return [[0], [1], [1], [1]]


# 8 20 1
class Perceptron:
    def __init__(self, layers, speed, alpha=1, func=None, default_weight=None, bias=True, propagate_method='stochastic',
                 save_results=False, continue_exists=None):
        if func is None:
            self.act_funcs = [self.sigmoid] * (len(layers) - 1)
        elif type(func) == str:
            if func in self.available_af:
                self.act_funcs = [self.__activate_functions[func]] * (len(layers) - 1)
            else:
                raise Exception('Incorrect activate function')
        elif type(func) == list:
            self.act_funcs = [self.__activate_functions[func[i]] for i in range((len(layers) - 1))]
        self.propagate_method = propagate_method if propagate_method in ('stochastic', 'batch') else 'stochastic'
        self.layers = layers
        self.async_loop = asyncio.get_event_loop()
        self.speed = speed
        self.alpha = alpha
        self.log = {'epochs': 0, 'learning_time': 0}
        self.bias = bias
        self.save_results = save_results
        self.network = []
        for i in range(1, len(layers)):
            layer = []
            for n in range(layers[i]):
                layer.append(
                    {"weights": np.array([default_weight or random() for _ in range(layers[i - 1] + (1 if bias else 0))])})
                if self.propagate_method == 'batch':
                    layer[-1]['batch_weights'] = np.array([0. for _ in range(layers[i - 1] + (1 if bias else 0))])
            self.network.append(layer)

    def calculate(self, inp):
        inputs = inp + [1]
        for ind, layer in enumerate(self.network):
            for neuron in layer:
                neuron['output'] = self.act_funcs[ind](sum([inputs[i] * w for i, w in enumerate(neuron['weights'])]))
            inputs = [neuron['output'] for neuron in layer] + [1]
        return [neuron['output'] for neuron in self.network[-1]]

    def __backward_propagation(self, d, res):
        if self.propagate_method == 'stochastic':
            self.__propagate(res)
            self.__update_weights(d)
        elif self.propagate_method == 'batch':
            self.__propagate(res)
            self.__acc_batch_weights(d)

    def __propagate(self, res):
        for i, neuron in enumerate(self.network[-1]):
            neuron['delta'] = self.act_funcs[-1](neuron['output'], True) * (res[i] - neuron['output'])
        for i, layer in enumerate(self.network[:-1][::-1]):
            for n, neuron in enumerate(layer):
                neuron['delta'] = self.act_funcs[-i - 2](neuron['output'], True) * sum(
                    [neuron['weights'][n] * neuron['delta'] for neuron in self.network[-i - 1]])

    def __update_weights(self, data):
        inputs = data
        for layer in self.network:
            for neuron in layer:
                for i in range(len(inputs)):
                    neuron['weights'][i] += self.speed * neuron['delta'] * inputs[i]
                neuron['weights'][-1] += self.speed * neuron['delta']
            inputs = [neuron['output'] for neuron in layer]

    def __acc_batch_weights(self, data):
        inputs = data
        for layer in self.network:
            for neuron in layer:
                for i in range(len(inputs)):
                    neuron['batch_weights'][i] += self.speed * neuron['delta'] * inputs[i]
                neuron['batch_weights'][-1] += self.speed * neuron['delta']
            inputs = [neuron['output'] for neuron in layer]

    def __update_batch_weights(self):
        for layer in self.network:
            for neuron in layer:
                neuron['weights'] += neuron['batch_weights']
                neuron['batch_weights'] = np.array([0. for _ in range(len(neuron['batch_weights']))])

    def __iteration(self, d, res):
        ans = self.calculate(d)
        self.__backward_propagation(d, res)
        return self.mse(ans, res)

    def __epoch(self, sel, res):
        error = 0
        for i, elem in enumerate(sel):
            error += self.__iteration(elem, res[i])
        if self.propagate_method == 'batch':
            self.__update_batch_weights()
        self.log["epochs"] += 1
        return error

    def teach(self, sel, res, epochs=None, accuracy=0.1):
        st = time()
        if epochs is None:
            err = self.__epoch(sel, res)
            while err > accuracy:
                err = self.__epoch(sel, res)
        else:
            for i in range(epochs):
                # print('Эпоха', self.log['epochs'])
                self.__epoch(sel, res)
        self.log['learning_time'] += time() - st

    def print(self, data, norm=lambda x: x, func=lambda x: x):
        from TableDraw import Table
        t = Table(f'Epochs: {self.log["epochs"]}\nLearning Time {round(self.log["learning_time"], 2)} sec\n'
                  f'Propagate method: {self.propagate_method}',
                  ['dataset', str(norm), str(func)])
        for row in data:
            t.insert_row([row, list(map(norm, self.calculate(row))), list(map(func, [self.calculate(row)]))])
        t.print_table()

    @property
    def __activate_functions(self):
        return {"threshold": self.threshold, "sigmoid": self.sigmoid, "tanh": self.tanh, "relu": self.relu,
                "leaky_relu": self.leaky_relu}

    @property
    def available_af(self):
        return ', '.join([elem for elem in self.__activate_functions])

    @staticmethod
    def threshold(inp):
        return int(inp >= 0.5)

    @staticmethod
    def sigmoid(x, derivative=False):
        return 1.0 / (1.0 + np.exp(-x)) if not derivative else (1.0 - x) * x

    @staticmethod
    def tanh(x, derivative=False):
        return (np.exp(2 * x) - 1) / (np.exp(2 * x) + 1) if not derivative else 1 - x ** 2

    @staticmethod
    def relu(x, derivative=False):
        return max(0, x) if not derivative else (0 if x <= 0 else 1)

    @staticmethod
    def leaky_relu(x, derivative=False):
        return (x * 0.01 if x < 0 else x) if not derivative else (0.01 if x < 0 else 1)

    @staticmethod
    def mse(t, p):
        return sum([(elem - p[i]) ** 2 for i, elem in enumerate(t)]) / len(t)

    @staticmethod
    def root_mse(t, p):
        return Perceptron.mse(t, p) ** 0.5

    @staticmethod
    def arctan(t, p):
        return sum([atan(elem - p[i]) ** 2 for i, elem in enumerate(t)]) / len(t)


if __name__ == '__main__':
    seed(666)
    functions = Selection()

    print('XOR')
    neural_network = Perceptron(layers=[2, 2, 1], func=['tanh', 'tanh'], speed=0.4, bias=True, propagate_method='stochastic')
    print(neural_network.network)
    neural_network.teach(functions.two_args, functions.xor_, epochs=150)
    neural_network.print(functions.two_args, norm=neural_network.threshold)

    seed(666)
    print('XOR')
    neural_network = Perceptron(layers=[2, 2, 1], func=['tanh', 'tanh'], speed=0.4, bias=True, propagate_method='batch')
    neural_network.teach(functions.two_args, functions.xor_, epochs=150)
    neural_network.print(functions.two_args, norm=neural_network.threshold)

    exit()

    print('\nBinary To Decimal | 1 output (float number)')
    neural_network = Perceptron(layers=[4, 6, 5, 1], speed=0.5, bias=True)
    neural_network.teach(functions.four_args, functions.bin_four, epochs=8000)
    neural_network.print(functions.four_args, norm=lambda x: round(x * 15))

    print('\nBinary To Decimal | 16 outputs (most similar)')
    neural_network = Perceptron(layers=[4, 6, 4, 16], speed=0.5, bias=True)
    neural_network.teach(functions.four_args, functions.bin_four_seq, epochs=2500)
    neural_network.print(functions.four_args, norm=lambda x: round(x, 1), func=lambda x: x.index(max(x)))

    """    for i, layer in enumerate(neural_network.network):
        print(f'LAYER {i}')
        for j, neuron in enumerate(layer):
            print(f'--NEURON {j}')
            print(f'----WEIGHTS {neuron["weights"]}')
            print(f'----OUTPUTS {neuron["output"]}')
            print(f'----DELTA {neuron["delta"]}')

"""
