import activation_functions as af

import numpy as np

class Perceptron:
    def __init__(self, activation_function, learning_rate):
        self.weights = np.random.uniform(-1, 1, 2)
        self.af = activation_function
        self.learning_rate = learning_rate
        self.bias = 1

    def forward(self, inputs):
        self.output = [np.dot(inputs, self.weights.T) + self.bias]
        self.af.forward(self.output)
        self.output = self.af.output

    def train(self, training_data):
        for point, classification in training_data:
            self.forward(point)
            error = classification - self.output[0]
            self.weights[0] += point[0] * error * self.learning_rate
            self.weights[1] += point[1] * error * self.learning_rate

    def test(self, test_data):
        correct = 0
        incorrect = 0
        for point, classification in test_data:
            self.forward(point)
            if(self.output[0] == classification):
                correct += 1
            else:
                incorrect += 1

        return (correct, incorrect)


if __name__ == "__main__":
    _af = af.ActivationFunction("step")
    perceptron = Perceptron(_af, .1)

    data_set = np.random.uniform(-1, 1, (1000, 2)) * 100

    def line(x,
             slope = np.random.uniform(-20, 20, 1)[0],
             intercept = np.random.uniform(-50, 50, 1)[0]
             ):
        return slope * x + intercept

    def data_classification(data, line):
        '''
        data set is a collection of 2 dimensional points.
        Classifies each point on whether they lay above or
        below a line defined by the values of slope and
        intercept.

        if below line, class=1
        if above or on line, class=0
        '''
        classified_data = []
        for point in data:
            classification = [point]
            if point[1] > line(point[0]):
                classification.append(1)
            else:
                classification.append(0)
            classified_data.append(classification)
        return classified_data

    classified_data = data_classification(data_set, line)

    correct, incorrect = perceptron.test(classified_data[800:])
    print("========Before Training=======")
    print(f"{(correct / (correct + incorrect)) * 100}% accurate")
    print(f"Correct: {correct}")
    print(f"Incorrect: {incorrect}")
    perceptron.train(classified_data[:799])
    correct, incorrect = perceptron.test(classified_data[800:])
    print("========After Training=======")
    print(f"{(correct / (correct + incorrect)) * 100}% accurate")
    print(f"Correct: {correct}")
    print(f"Incorrect: {incorrect}")

