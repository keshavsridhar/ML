import numpy as np
import random as rd
import matplotlib.pyplot as plot


class Neuron:
    '''
    Neuron class:
    Functions: summate(), activation()
    > summate(self): Takes the dot product of input and the weights and adds bias term and stores in self.net_input
    > activation(self): Returns a sigmoid activation of the net_input and stores in self.output
    '''

    def __init__(self, inpsize):
        self.bias = rd.uniform(-1, 1)
        self.weight = np.random.randn(inpsize)
        self.net_input = 0
        self.output = 0
        self.delta = 0
        # These two terms are needed for calculating momentum
        self.old_bias = 0
        self.old_weight = 0

    def summate(self, inp):
        self.net_input = np.dot(inp, self.weight) + self.bias

    def activation(self):
        self.output = 1.0/(1.0 + np.exp(-self.net_input))


class NeuralNet:
    '''
    Neural network class:
    Functions: train_sgd(), back_propagate(), forward_propagate(), predict()
    train_sgd(): Invokes stochastic gradient descent and calls other functions
    forward_propagate(): Invokes the forward propagation
    back_propagate(): Invokes the back propagation as well as weight & bias updates
    predict(): Predicts on data after training is done.
    '''
    def __init__(self, x, y, hd, out, e):
        self.data = np.array(x, dtype="float")
        self.label = np.array(y, dtype="float")
        self.hidden_layer = [Neuron(self.data.shape[1]) for i in range(hd)]
        self.output_layer = [Neuron(len(self.hidden_layer)) for i in range(out)]
        self.delta_output_layer = 0
        self.eta = e
        self.error_rates = []

    def train_sgd(self, nepoch):
        for epoch in range(nepoch):
            sum_error = 0
            classified_count = 0
            # Stochastic gradient descent:
            for row, y_true in zip(self.data, self.label):
                # Propagate forward to calculate output:
                prev_out = self.forward_propagate(row)
                # Propagate backward to update weights (without momentum term):
                # self.back_propagate(y_true, prev_out, row)
                # Back propagate with momentum term:
                self.back_propagate_with_momentum(y_true, prev_out, row)
                # Calculate error for this data point at the output layer:
                for neur in self.output_layer:
                    # absolute error:
                    if round(float(abs(neur.output - y_true)), 3) <= 0.05:
                        classified_count += 1
                    # absolute error sum:
                    sum_error += abs(neur.output - y_true)
            if classified_count == 16:
                break
            print("Epoch: {0}, Sum_error: {1}, classified_count: {2}".format(epoch, sum_error, classified_count))
            self.error_rates.append(sum_error)

    def back_propagate(self, y, o1, r):
        for neur in self.output_layer:
            # Output delta:
            neur.delta = neur.output * (1 - neur.output) * (y - neur.output)
            out_weights = neur.weight*neur.delta
            # Weight-update for output layer:
            neur.bias += self.eta*neur.delta
            neur.weight += self.eta*neur.delta * o1

        for neur1, i, j in zip(self.hidden_layer, out_weights, r):
            # Hidden layer delta:
            neur1.delta = neur1.output * (1 - neur1.output) * i * j
            # Weight-update for hidden layer:
            neur1.bias += self.eta*neur1.output * (1 - neur1.output) * i
            neur1.weight += self.eta*neur1.delta
        return

    def forward_propagate(self, train_row):
        hidden_out = []
        for neur in self.hidden_layer:
            neur.summate(train_row)
            neur.activation()
            hidden_out.append(neur.output)
        for neur1 in self.output_layer:
            neur1.summate(hidden_out)
            neur1.activation()
        return hidden_out

    def predict(self, x):
        # Predict 1 if output layer activation >= 0.5 else predict 0
        predicted = []
        for row in x:
            interim = self.forward_propagate(row)
            for neur1 in self.output_layer:
                neur1.summate(interim)
                neur1.activation()
                print(neur1.output)
                if neur1.output >= 0.5:
                    predicted.append([1])
                else:
                    predicted.append([0])
        return predicted

    def back_propagate_with_momentum(self, y, o1, r):
        for neur in self.output_layer:
            # Output delta:
            neur.delta = neur.output * (1 - neur.output) * (y - neur.output)
            # print("y_true: {0}, delta: {1}, output_layer: {2}".format(y, neur.delta, neur.output))
            out_weights = neur.weight*neur.delta
            # Weight-update for output layer:
            neur.bias += (self.eta*neur.delta) + 0.9*neur.old_bias
            neur.weight += (self.eta*neur.delta * o1) + 0.9*neur.old_weight
            neur.old_bias = self.eta*neur.delta
            neur.old_weight = self.eta*neur.delta * o1
            # print("Output bias: {0}, Output weight: {1}".format(neur.bias, neur.weight))

        for neur1, i, j in zip(self.hidden_layer, out_weights, r):
            # Hidden layer delta:
            neur1.delta = neur1.output * (1 - neur1.output) * i * j
            # Weight-update for hidden layer:
            neur1.bias += (self.eta*neur1.output * (1 - neur1.output) * i) + 0.9*neur1.old_bias
            neur1.weight += (self.eta*neur1.delta) + 0.9*neur1.old_weight
            neur1.old_bias = self.eta*neur1.output * (1 - neur1.output) * i
            neur1.old_weight = self.eta*neur1.delta
            # print("Output bias: {0}, Output weight: {1}".format(neur1.bias, neur1.weight))
        return

if __name__ == "__main__":
    # Train data:
    trainX = [[0, 0, 0, 0],
              [0, 0, 0, 1],
              [0, 0, 1, 0],
              [0, 0, 1, 1],
              [0, 1, 0, 0],
              [0, 1, 0, 1],
              [0, 1, 1, 0],
              [0, 1, 1, 1],
              [1, 0, 0, 0],
              [1, 0, 0, 1],
              [1, 0, 1, 0],
              [1, 0, 1, 1],
              [1, 1, 0, 0],
              [1, 1, 0, 1],
              [1, 1, 1, 0],
              [1, 1, 1, 1]]
    # Train label (parity):
    trainy = [[0], [1], [1], [0], [1], [0], [0], [1], [1], [0], [0], [1], [0], [1], [1], [0]]

    # The last parameter in the below call is the "eta" parameter:
    # Initialize network:
    net = NeuralNet(trainX, trainy, 4, 1, 0.10)
    # Train the network (Set epoch number as needed):
    net.train_sgd(750000)
    # Get the predictions:
    preds = net.predict(trainX)
    print("Train y's: {0}".format(trainy))
    print("Predicted y's: {0}".format(preds))
    # Plot absolute sum error against the number of epochs:
    plot.style.use("ggplot")
    plot.title("Absolute error vs # of epochs (Eta: 0.10):")
    plot.ylabel("Absolute error sum")
    plot.xlabel("Epoch number")
    plot.scatter(range(len(net.error_rates)), net.error_rates, marker=".", c="Black")
    plot.show()
