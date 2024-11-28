class SimplePerceptron:
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.weights = []
        self.bias = 0

    def activation_function(self, net):
        # Step 6: Step function for activation
        return 1 if net >= 0 else 0

    def fit(self, X, y, max_epochs=1000):
        # Step 1: Initialize weights and bias
        self.weights = [0] * len(X[0])  # Number of features
        self.bias = 0

        for epoch in range(max_epochs):
            errors = 0

            # Step 3: For each training pair s:t
            for i in range(len(X)):
                # Step 4: Set activations of input units ai = xi
                ai = X[i]

                # Step 5: Calculate the summing part value Net = Σ ai * wi + b
                net = sum(ai[j] * self.weights[j] for j in range(len(ai))) + self.bias

                # Step 6: Compute the response of output unit
                output = self.activation_function(net)

                # Step 7: Update weights and bias if an error occurred
                if output != y[i]:
                    errors += 1
                    for j in range(len(self.weights)):
                        self.weights[j] += self.learning_rate * (y[i] - output) * ai[j]
                    self.bias += self.learning_rate * (y[i] - output)

            # Step 8: Test Stopping Condition
            if errors == 0:
                break

    def predict(self, X):
        predictions = []
        for ai in X:
            net = sum(ai[j] * self.weights[j] for j in range(len(ai))) + self.bias
            predictions.append(self.activation_function(net))
        return predictions


# Gate selection and training
def get_gate_data(gate_type):
    if gate_type == "AND":
        return [[0, 0], [0, 1], [1, 0], [1, 1]], [0, 0, 0, 1]
    elif gate_type == "OR":
        return [[0, 0], [0, 1], [1, 0], [1, 1]], [0, 1, 1, 1]
    else:
        print("Invalid gate type. Please choose from AND or OR.")
        return None, None


if __name__ == "__main__":
    while True:
        # Ask the user which gate to solve
        gate_type = input("Enter the gate you want to solve (AND, OR) or type 'quit' to exit: ").strip().upper()

        if gate_type == 'QUIT':
            print("Exiting program.")
            break

        # Get the data for the selected gate
        X, y = get_gate_data(gate_type)

        if X and y:
            perceptron = SimplePerceptron(learning_rate=0.1)
            perceptron.fit(X, y)

            # Test the perceptron
            predictions = perceptron.predict(X)
            print(f"Predictions for {gate_type} gate:", predictions)

        # Ask if the user wants to solve another gate or quit
        continue_choice = input("Do you want to solve another gate? (yes/no): ").strip().lower()
        if continue_choice != 'yes':
            print("Exiting program.")
            break
