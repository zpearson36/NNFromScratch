import activation_functions as af

import math
import numpy as np


class MLPerceptron:

    def __init__(self, num_inputs, num_hidden, num_output, activation_function):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_output = num_output
        self.input_weights = np.random.uniform(-1, 1, (self.num_inputs, self.num_hidden))
        #self.output_weights = np.full((self.num_hidden, self.num_output), 1)
        self.output_weights = np.random.uniform(-1, 1, (self.num_hidden, self.num_output))
        self.input_biases = np.full((self.num_hidden), 1)
        self.output_biases = np.full(1, (self.num_output))
        self.a_function = activation_function
        self.learning_rate = 1

    def forward(self, inputs):
        assert len(inputs) == self.num_inputs
        self.a_function.forward(
            np.dot(self.input_weights.T, np.array(inputs)) + self.input_biases
            )
        self.output = self.a_function.step_function(
                np.dot(self.output_weights.T,
                       np.array(self.a_function.output).reshape((self.num_hidden,1))
                       ) + self.output_biases
                )

    def train(self, training_data):
        for inputs, classification in training_data:
            self.forward(inputs)
            errors = np.array(classification) - np.array(self.output)

            # e_list = np.divide(np.multiply(self.output_weights, error), self.output_weights.sum())
            # self.input_weights += np.multiply(np.dot(self.input_weights, e_list), self.learning_rate)
            # self.output_weights += np.multiply(np.dot(self.a_function.output, self.output_weights), self.learning_rate)

            e_list = []
            for weight in self.output_weights:
                e_val = 0
                for w, error in zip(weight, errors):
                    e_val += (w / self.output_weights.sum()) * error
                e_list.append(e_val)

            for inpt, weights in zip(inputs, self.input_weights):
                for error, weight in zip(e_list, weights):
                    x = np.multiply(np.multiply(inpt, error), self.learning_rate)
                    weight += x


            for weight, output in zip(self.output_weights, self.a_function.output):
                weight += np.multiply(np.multiply(output, error), self.learning_rate)
            

    def test(self, test_data):
        correct = 0
        incorrect = 0
        for point, classification in test_data:
            self.forward(point)
            matches = True
            for output, _class in zip(self.output, classification):
                matches = False if output != _class else True

            if(matches):
                correct += 1
            else:
                incorrect += 1

        return (correct, incorrect)


if __name__ == "__main__":
    #afunction = af.ActivationFunction("rectifiedlinear")
    afunction = af.ActivationFunction("sigmoid")
    ml = MLPerceptron(2, 40, 1, afunction)

    data_set = np.random.uniform(-1, 1, (1000, 2)) * 100

    degree = np.random.randint(1, 20) 
    coeffs = np.random.uniform(-100, 100, (degree))

    def f(x, coeffs):
        y = 0
        for power, coeff in enumerate(coeffs):
            y += coeff * x ** power

        return y

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
            if point[1] > curve(point[0], coeffs):
                classification.append([1])
            else:
                classification.append([0])
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
