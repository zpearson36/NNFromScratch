import activation_functions as af
import MLperceptron as mlp

if __name__ == '__main__':
    a_function = af.ActivationFunction("sigmoid")
    brain = mlp.MLPerceptron(64, 64, 10, a_function)

    data_set = []

    with open('.ignore/optdigits.tes') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip('\n')
        line = line.split(",")
        line = [eval(i) for i in line]
        digit = line.pop(-1)
        _class = [0] * 10
        _class[digit] = 1
        data_set.append((line, _class))

    print(len(data_set))
    correct, incorrect = brain.test(data_set[1000:])
    print("========Before Training=======")
    print(f"{(correct / (correct + incorrect)) * 100}% accurate")
    print(f"Correct: {correct}")
    print(f"Incorrect: {incorrect}")
    brain.train(data_set[:999])
    correct, incorrect = brain.test(data_set[1000:])
    print("========After Training=======")
    print(f"{(correct / (correct + incorrect)) * 100}% accurate")
    print(f"Correct: {correct}")
    print(f"Incorrect: {incorrect}")

