import activation_functions as af

import numpy as np

class Perceptron:
    def __init__(self, activation_function, learning_rate):
        self.weights = np.empty(2)
        self.af = activation_function
        self.learning_rate = learning_rate

    def forward(self, inputs):
        self.output = [np.dot(inputs, self.weights.T)]
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

    data_set = np.random.random((1000, 2)) * 100

    def data_classification(data):
        classified_data = []
        for point in data:
            classification = [point]
            if point[0] > point[1]:
                classification.append(1)
            else:
                classification.append(0)
            classified_data.append(classification)
        return classified_data

    classified_data = data_classification(data_set)

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

