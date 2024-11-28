import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Initialize weights and biases with small random values
        self.weights_input_hidden = np.random.uniform(-0.5, 0.5, (self.input_size, self.hidden_size))
        self.weights_hidden_output = np.random.uniform(-0.5, 0.5, (self.hidden_size, self.output_size))
        self.bias_hidden = np.random.uniform(-0.5, 0.5, self.hidden_size)
        self.bias_output = np.random.uniform(-0.5, 0.5, self.output_size)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.sigmoid(self.hidden_input)
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.final_output = self.sigmoid(self.final_input)
        return self.final_output

    def backward(self, X, y, output):
        output_error = y - output
        output_delta = output_error * self.sigmoid_derivative(output)

        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_output)

        self.weights_hidden_output += self.hidden_output.T.dot(output_delta) * self.learning_rate
        self.bias_output += np.sum(output_delta, axis=0) * self.learning_rate
        self.weights_input_hidden += X.T.dot(hidden_delta) * self.learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0) * self.learning_rate

    def train(self, X, y, epochs=10000):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)
            loss = np.mean(np.square(y - output))
            if epoch % 1000 == 0 or loss < 0.01:  # Periodic loss updates
                print(f"Epoch {epoch}: Loss = {loss:.4f}")
            if loss < 0.01:  # Early stopping
                print(f"Early stopping at epoch {epoch} with loss {loss:.4f}")
                break

    def predict(self, X):
        return self.forward(X)

def main():
    print("Training a Neural Network for XOR gate.")

    # Get XOR gate inputs and outputs from the user
    num_pairs = int(input("Enter the number of input-output pairs for XOR gate: "))
    X = []
    y = []

    for i in range(num_pairs):
        pair = input(f"Enter input-output pair {i + 1} (e.g., 0 1 1 for input [0, 1] and output 1): ")
        pair_values = list(map(int, pair.split()))
        X.append(pair_values[:-1])  # Inputs (all values except the last one)
        y.append([pair_values[-1]])  # Output (the last value)

    X = np.array(X)
    y = np.array(y)

    hidden_size = int(input("Enter the number of neurons in the hidden layer: "))
    epochs = int(input("Enter the number of epochs for training: "))

    input_size = X.shape[1]
    output_size = y.shape[1]
    learning_rate = 0.1

    nn = NeuralNetwork(input_size, hidden_size, output_size, learning_rate)
    nn.train(X, y, epochs=epochs)

    print("\nTraining complete.")

    print("\nTesting the model with the input data:")
    predictions = nn.predict(X)
    rounded_predictions = np.round(predictions)

    print("\nTest Results:")
    for i, prediction in enumerate(rounded_predictions):
        print(f"Input: {X[i]} => Predicted Output: {int(prediction[0])} (Expected: {y[i][0]})")

if __name__ == "__main__":
    main()
