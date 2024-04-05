import activation_functions as af

import json
import math
import numpy as np
import os


class MLPerceptron:

    def __init__(self, num_inputs, num_hidden, num_output, activation_function):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_output = num_output
        self.input_weights = np.random.uniform(-1, 1, (self.num_inputs, self.num_hidden))
        self.output_weights = np.random.uniform(-1, 1, (self.num_hidden, self.num_output))
        self.input_biases = np.full((self.num_hidden), 1)
        self.output_biases = np.full(1, (self.num_output))
        self.a_function = activation_function
        self.learning_rate = .01

    def forward(self, inputs):
        assert len(inputs) == self.num_inputs
        self.a_function.forward(
            np.dot(self.input_weights.T, np.array(inputs)) + self.input_biases
            )
        a = self.output_weights.T
        b = np.array(self.a_function.output).reshape((self.num_hidden,1))
        #print(a.shape)
        #print(b.shape)
        #print(np.dot(a, b).shape)
        #print("="*20)
        self.output = self.a_function.rect_linear(
                np.dot(self.output_weights.T,
                       np.array(self.a_function.output).reshape((self.num_hidden,1))
                       ) + self.output_biases
                )

    def train(self, training_data):
        for inputs, classification in training_data:
            self.forward(inputs)
            errors = np.array(classification) - np.array(self.output)
            print(np.array(classification))
            print(self.output)
            print(errors)

            e_list = np.divide(np.multiply(self.output_weights, errors), self.output_weights.sum())
            self.output_weights += np.multiply(e_list, self.learning_rate)

            e_list = np.dot(np.array(errors), self.output_weights.T)
            e_list = np.divide(np.multiply(self.input_weights, e_list), self.input_weights.sum())
            self.input_weights += np.multiply(e_list, self.learning_rate)

    def test(self, test_data):
        correct = 0
        incorrect = 0
        for point, classification in test_data:
            self.forward(point)
            matches = True
            index = 0

            for indx, val in enumerate(self.output):
                if(val > self.output[indx]):
                    index = indx
            prediction = [0] * len(self.output)
            prediction[index] = 1
            #print(f"{prediction}:{self.output}")

            for output, _class in zip(prediction, classification):
                matches = False if output != _class else True

            if(matches):
                correct += 1
            else:
                incorrect += 1

        return (correct, incorrect)

    def save_weights(self, file):
        weights_dict = {}
        weights_dict["input"] = str(self.input_weights.tolist())
        weights_dict["output"] = str(self.output_weights.tolist())
        with open(file, "w") as f:
            json.dump(weights_dict, f)

        print(f"Weights Saved to {file}")

    def load_weights(self, file):
        with open(file, "r") as  f:
            weights_json = json.load(f)
        input_weights  = weights_json["input"].split("], [")
        input_weights  = [weight.replace("[", "").replace("]", "") for weight in input_weights]
        input_weights2 = []

        for weights in input_weights:
            input_weights2.append([float(weight) for weight in weights.split(', ')])

        self.input_weights = np.array(input_weights2)

        output_weights  = weights_json["output"].split("], [")
        output_weights  = [weight.replace("[", "").replace("]", "") for weight in output_weights]
        output_weights2 = []

        for weights in output_weights:
            output_weights2.append([float(weight) for weight in weights.split(', ')])

        self.output_weights = np.array(output_weights2)

        print("Weights loaded")


if __name__ == "__main__":

    hidden_layer_size = 40
    function = "x2"

    afunction = af.ActivationFunction("rectifiedlinear")
    ml = MLPerceptron(2, hidden_layer_size, 2, afunction)

    data_set = np.random.uniform(-1, 1, (1000, 2)) * 100
    weights_path = f".ignore/weights_{function}_{hidden_layer_size}"
    if(os.path.isfile(weights_path)):
        ml.load_weights(weights_path)

    def f(x):
        return x**2

    def data_classification(data, curve):
        '''
        data set is a collection of 2 dimensional points.
        Classifies each point on whether they lay above or
        below an arbitrary curve defined by the given function.

        if below curve, class=0
        if above or on line, class=1
        '''
        classified_data = []
        for point in data:
            classification = [point]
            if point[1] > curve(point[0]):
                classification.append([1, 0])
            else:
                classification.append([0, 1])
            classified_data.append(classification)
        return classified_data

    classified_data = data_classification(data_set, f)
    correct, incorrect = ml.test(classified_data[800:])
    print("========Before Training=======")
    print(f"{(correct / (correct + incorrect)) * 100}% accurate")
    print(f"Correct: {correct}")
    print(f"Incorrect: {incorrect}")
    ml.train(classified_data[:799])
    correct, incorrect = ml.test(classified_data[800:])
    print("========After Training=======")
    print(f"{(correct / (correct + incorrect)) * 100}% accurate")
    print(f"Correct: {correct}")
    print(f"Incorrect: {incorrect}")
    if((correct / (correct + incorrect)) >= .9):
        ml.save_weights(weights_path)
