import math

import nnfs
from nnfs.datasets import spiral_data
import numpy as np


class ActivationFunction:
    '''
    Class to contain all Activation Functions
    '''
    def __init__(self, _type):
        match _type:
            case "step":
                self.function = self.step_function
            case "sigmoid":
                self.function = self.sigmoid_function
            case "rectifiedlinear":
                self.function = self.rect_linear
            case _:
                raise ValueError

    def forward(self, inputs):
        self.output = self.function(inputs)
        return self.output

    def step_function(self, inputs):
        output = []
        for i in inputs:
            if i > 0:
                output.append(1)
            else:
                output.append(0)
        return output
    
    def sigmoid_function(self, inputs):
        output = []
        for i in inputs:
            i = np.clip(i, -500, 500)
            output.append(1 / (1 + math.exp(-i)))
        return output
    
    def rect_linear(self, inputs):
        output = []
        for i in inputs:
            output.append(np.maximum(0, inputs)[0])
        return output

if __name__ == '__main__':
    nnfs.init()
    X = [
          [1, 2, 3, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]
             ]

    X, y = spiral_data(100,3)

    inputs = [0.0, 2.0, -1, 3.3, -2.7,1.1, 2.2, -100]

    output1 = rect_linear(inputs)
    output2 = sigmoid_function(inputs)
    output3 = step_function(inputs)

    print(output1)
    print(output2)
    print(output3)
