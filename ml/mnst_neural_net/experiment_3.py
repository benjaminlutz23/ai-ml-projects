# Benjamin Lutz
# Training a Neural Network on the MNIST data set


import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of the sigmoid function
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Load and preprocess the MNIST dataset
def load_data(sample_fraction):
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images.reshape((train_images.shape[0], -1))
    test_images = test_images.reshape((test_images.shape[0], -1))
    
    # Sample a fraction of the training data
    num_samples = int(sample_fraction * len(train_images))
    indices = np.random.permutation(len(train_images))[:num_samples]
    train_images = train_images[indices]
    train_labels = train_labels[indices]

    return (train_images, train_labels), (test_images, test_labels)

# Preprocess the data
def preprocess(images, labels):
    images = images.astype("float32") / 255.0
    labels = to_categorical(labels, 10)
    labels[labels == 1] = 0.9
    labels[labels == 0] = 0.1
    return images, labels

# Neural network class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1, momentum=0.9):
        self.input_size = input_size + 1  # Adjust input size for bias
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.init_weights()

    def init_weights(self):
        self.weights_input_hidden = np.random.uniform(-0.05, 0.05, (self.input_size, self.hidden_size))
        self.weights_hidden_output = np.random.uniform(-0.05, 0.05, (self.hidden_size + 1, self.output_size))
        self.velocity_input_hidden = np.zeros_like(self.weights_input_hidden)
        self.velocity_hidden_output = np.zeros_like(self.weights_hidden_output)

    def forward_pass(self, inputs):
        self.inputs = np.append(inputs, 1)
        self.hidden_inputs = np.dot(self.inputs, self.weights_input_hidden)
        self.hidden_outputs = sigmoid(self.hidden_inputs)
        self.hidden_outputs = np.append(self.hidden_outputs, 1)
        self.final_inputs = np.dot(self.hidden_outputs, self.weights_hidden_output)
        self.final_outputs = sigmoid(self.final_inputs)
        return self.final_outputs

    def backward_pass(self, targets):
        output_errors = targets - self.final_outputs
        hidden_errors = np.dot(output_errors, self.weights_hidden_output[:-1].T) * sigmoid_derivative(self.hidden_inputs)
        self.velocity_hidden_output = self.momentum * self.velocity_hidden_output + self.learning_rate * np.outer(self.hidden_outputs, output_errors)
        self.weights_hidden_output += self.velocity_hidden_output
        self.velocity_input_hidden = self.momentum * self.velocity_input_hidden + self.learning_rate * np.outer(self.inputs, hidden_errors)
        self.weights_input_hidden += self.velocity_input_hidden

    def train(self, training_data, training_targets, test_data, test_targets, epochs=50):
        train_accuracies = []
        test_accuracies = []
        for epoch in range(epochs):
            for (inputs, target) in zip(training_data, training_targets):
                self.forward_pass(inputs)
                self.backward_pass(target)
            train_accuracy, _ = self.evaluate(training_data, training_targets)
            train_accuracies.append(train_accuracy)
            test_accuracy, _ = self.evaluate(test_data, test_targets)
            test_accuracies.append(test_accuracy)
            print(f"Epoch {epoch + 1}, Training Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%")
        return train_accuracies, test_accuracies

    def evaluate(self, data, targets):
        predictions = []
        for inputs in data:
            outputs = self.forward_pass(inputs)
            predictions.append(np.argmax(outputs))
        actual = np.argmax(targets, axis=1)
        accuracy = (predictions == actual).mean() * 100
        return accuracy, predictions

    def plot_confusion_matrix(self, true_classes, predictions):
        cm = confusion_matrix(true_classes, predictions)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()

# Main function
def main():
    fractions = [0.25, 0.5]  # Fractions of the training data to use
    hidden_size = 100  # Fixed number of hidden units
    momentum = 0.9  # Fixed momentum value
    epoch_count = 50  # Number of training epochs

    for fraction in fractions:
        print(f"Training with {fraction * 100}% of the training data")
        (train_images, train_labels), (test_images, test_labels) = load_data(fraction)
        train_images, train_labels = preprocess(train_images, train_labels)
        test_images, test_labels = preprocess(test_images, test_labels)
        nn = NeuralNetwork(784, hidden_size, 10, momentum=momentum)
        train_accuracies, test_accuracies = nn.train(train_images, train_labels, test_images, test_labels, epochs=epoch_count)

        # Plot training and test accuracy over epochs
        plt.plot(range(epoch_count), train_accuracies, label="Training")
        plt.plot(range(epoch_count), test_accuracies, label="Test")
        plt.title(f"Training and Test Accuracy with {fraction * 100}% Training Data")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()

        # Evaluate and plot confusion matrix
        final_test_accuracy, test_predictions = nn.evaluate(test_images, test_labels)
        print(f"Final test accuracy with {fraction * 100}% training data: {final_test_accuracy:.2f}%")
        actual_classes = np.argmax(test_labels, axis=1)
        nn.plot_confusion_matrix(actual_classes, test_predictions)

if __name__ == "__main__":
    main()
