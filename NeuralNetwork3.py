import math
import pickle
import sys
import numpy as np
import random
# import time

output_filename = "test_predictions.csv"
trained_model_filename = "trained_model.pickle"
# test_labels_fp = "./data/small/test_label.csv"


def read_input():
    if len(sys.argv) > 1:
        train_images_fp = sys.argv[1]
        train_labels_fp = sys.argv[2]
        test_images_fp = sys.argv[3]
    else:
        train_images_fp = "./data/small/train_image.csv"
        train_labels_fp = "./data/small/train_label.csv"
        test_images_fp = "./data/small/test_image.csv"

    return train_images_fp, train_labels_fp, test_images_fp


def preprocess_data(train_images_fp, train_labels_fp, test_images_fp):
    x_train_raw = np.genfromtxt(train_images_fp, delimiter=",")
    y_train_raw = np.genfromtxt(train_labels_fp, delimiter=",")
    x_test_raw = np.genfromtxt(test_images_fp, delimiter=",")
    # y_test_raw = np.genfromtxt(test_labels_fp, delimiter=",")

    x_train = [np.reshape(x/255, (784, 1)) for x in x_train_raw]
    y_train = [vectorized_result(int(y)) for y in y_train_raw]
    training_data = zip(x_train, y_train)

    x_test = [np.reshape(x/255, (784, 1)) for x in x_test_raw]
    # test_data = zip(x_test, y_test_raw)

    return list(training_data), list(x_test)


def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


def sigmoid(input_vec):
    return 1/(1 + np.exp(-input_vec))


def derivative_sigmoid(input_vec):
    return sigmoid(input_vec) * (1 - sigmoid(input_vec))


# def softmax(x):
#     return np.exp(x) / np.sum(np.exp(x), axis=0)

def stable_softmax(x):
    return np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)))


def cross_entropy(y_pred, y_true):
    return -sum([y_true[i]*math.log(y_pred[i]) for i in range(len(y_pred))])


class NNModel:
    def __init__(self, input_dim, hidden_layers, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weights = []
        self.biases = []

        for idx in range(len(hidden_layers) + 1):
            b_out = output_dim if idx == len(hidden_layers) else hidden_layers[idx]
            # bias = np.random.rand(b_out, 1) * np.sqrt(1 / (b_out + 1))
            bias = np.random.randn(b_out, 1)
            self.biases.append(bias)

            n_in = input_dim if idx == 0 else hidden_layers[idx - 1]
            n_out = output_dim if idx == len(hidden_layers) else hidden_layers[idx]
            # wt = np.random.rand(n_out, n_in) * np.sqrt(1 / (n_out + n_in))
            wt = np.random.randn(n_out, n_in) / np.sqrt(n_in)
            self.weights.append(wt)

    def train_neural_network(self, training_data, learning_rate, epochs, batch_size):
        # train nn for #epochs
        for epoch in range(epochs):
            loss = 0
            random.shuffle(training_data)
            # set batch derivatives to zeroes
            batch_dw = [np.zeros(weight.shape) for weight in self.weights]
            batch_db = [np.zeros(bias.shape) for bias in self.biases]

            # iterate all batches of the dataset
            for idx, input_vec in enumerate(training_data):
                # calculate forward pass
                y_pred, cached_outputs = self.forward_pass(input_vec[0])

                # calculate cross entropy loss
                loss += cross_entropy(y_pred, input_vec[1])

                # calculate backward propagation
                dw, db = self.back_propagation(input_vec, cached_outputs)

                # add derivatives to batch derivative
                batch_dw = np.add(batch_dw, dw, dtype="object")
                batch_db = np.add(batch_db, db, dtype="object")

                # update weights and bias when batch_size is met
                if idx == len(training_data)-1 or (idx + 1) % batch_size == 0:
                    curr_batch_size = batch_size if (idx + 1) % batch_size == 0 else (idx + 1) % batch_size
                    self.weights = np.subtract(self.weights, np.multiply(batch_dw, (learning_rate / curr_batch_size)))
                    self.biases = np.subtract(self.biases, np.multiply(batch_db, (learning_rate / curr_batch_size)))

                    # reset batch derivatives to zeroes
                    batch_dw = [np.zeros(weight.shape) for weight in self.weights]
                    batch_db = [np.zeros(bias.shape) for bias in self.biases]

            print("Average Loss ", epoch, ": ", loss/len(training_data))

    def forward_pass(self, input_vec):
        cached_outputs = dict()
        for layer_idx in range(len(self.weights)):
            input_vec = np.add(np.dot(self.weights[layer_idx], input_vec), self.biases[layer_idx])
            cached_outputs["z" + str(layer_idx)] = input_vec
            if layer_idx == len(self.weights)-1:
                input_vec = stable_softmax(input_vec)
            else:
                input_vec = sigmoid(input_vec)
            cached_outputs["a" + str(layer_idx)] = input_vec

        return input_vec, cached_outputs

    def back_propagation(self, input_vec, cached_outputs):
        dw = [np.zeros(weight.shape) for weight in self.weights]
        db = [np.zeros(bias.shape) for bias in self.biases]

        for idx in range(len(self.weights)-1, -1, -1):
            if idx == len(self.weights)-1:
                dz = cached_outputs["a" + str(idx)] - input_vec[1]
            else:
                dz = np.dot(self.weights[idx+1].T, dz) * derivative_sigmoid(cached_outputs["z" + str(idx)])
            if idx == 0:
                dw[idx] = np.dot(dz, input_vec[0].T)
            else:
                dw[idx] = np.dot(dz, cached_outputs["a" + str(idx-1)].T)
            db[idx] = dz
        return dw, db

    def test(self, test_data):
        # correct = 0
        with open(output_filename, "w") as file:
            for test_input in test_data:
                # pred, _ = self.forward_pass(test_input[0])
                pred, _ = self.forward_pass(test_input)
                pred = np.argmax(pred)
                file.write(str(int(pred)) + "\n")
                # if pred == test_input[1]:
                #     correct += 1
        # print("accuracy: ", correct*100/len(test_data))


def main():
    # start = time.time()
    # read and preprocess data
    train_images, train_labels, test_images = read_input()
    training_data, test_data = preprocess_data(train_images, train_labels, test_images)

    # initialise model
    model = NNModel(input_dim=784, hidden_layers=[200, 30], output_dim=10)
    # train model
    model.train_neural_network(training_data, learning_rate=0.2, epochs=15, batch_size=10)
    # get predictions for test data
    model.test(test_data)

    # write output
    with open(trained_model_filename, "wb") as pf:
        pickle.dump(model, pf)

    # end = time.time()
    # print(end - start)


if __name__ == '__main__':
    main()
