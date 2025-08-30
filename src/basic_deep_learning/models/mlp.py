"""Main module containing the MultiLayerPerceptron class."""
from typing import Literal
import matplotlib.pyplot as plt
import json
from ..functionality.matrix import Matrix
from ..functionality.activations_registry import ActivationFunctionsRegistry
import time
from datetime import datetime
import os

def time_format(seconds: float):
    floor_secs = int(seconds)
    h, s = divmod(floor_secs, 3600)
    m, sec = divmod(s, 60)
    return f'{h:02d} h : {m:02d} m : {sec:02d} s : {int((seconds-floor_secs)*1000)} ms'


class MultiLayerPerceptron:
    def __init__(self,
                structure: list[int],
                hidden_layer_activation_function_label: Literal['sigmoid', 'ReLU', 'linear', 'tanh'],
                output_layer_activation_function_label: Literal['sigmoid', 'ReLU', 'linear', 'tanh']):
        self.structure = structure
        self.number_of_layers = len(structure)
        self.weights = [
            Matrix.randomize(y, x, -1, 1) for x, y in zip(structure[:-1], structure[1:])
        ]
        self.biases = [
            Matrix.randomize(y, 1, -1, 1) for y in structure[1:]
        ]
        self.hidden_activation_name = hidden_layer_activation_function_label
        self.output_activation_name = output_layer_activation_function_label
        self.hidden_activation_function, self.hidden_activation_function_prime = ActivationFunctionsRegistry.Activations[
            hidden_layer_activation_function_label
            ]
        self.output_activation_function, self.output_activation_function_prime = ActivationFunctionsRegistry.Activations[
            output_layer_activation_function_label
            ]
        
    def forward_propagate(self, input_vector: Matrix):
        """Feeds forward an input vector into the MLP."""
        if input_vector.format[1] != 1:
            raise ValueError("Input must be a column vector")
        activations = [input_vector]
        pre_activations = []
        current_activation = input_vector
        for i, (W, B) in enumerate(zip(self.weights, self.biases)):
            Z = W * current_activation + B
            pre_activations.append(Z)
            if i < len(self.weights) - 1:
                current_activation = self.hidden_activation_function(Z)
            else:
                current_activation = self.output_activation_function(Z)
            activations.append(current_activation)
        return activations[-1], (activations, pre_activations)
    
    def backward_propagate(self, input_vector: Matrix, expected_vector: Matrix, learning_rate: float = 0.1):
        """Updates the weights and biases based on one input and its expected output."""
        _, (activations, pre_activations) = self.forward_propagate(input_vector)
        grad_w = [Matrix.zero(*W.format) for W in self.weights]
        grad_b = [Matrix.zero(*B.format) for B in self.biases]
        output_error = (activations[-1] - expected_vector) @ self.output_activation_function_prime(pre_activations[-1])
        grad_b[-1] = output_error
        grad_w[-1] = output_error * activations[-2].T()
        error = output_error
        for layer_idx in range(len(self.weights)-2, -1, -1):
            error = (self.weights[layer_idx+1].T() * error) @ self.hidden_activation_function_prime(pre_activations[layer_idx])
            grad_b[layer_idx] = error
            grad_w[layer_idx] = error * activations[layer_idx].T()
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] - (learning_rate * grad_w[i])
            self.biases[i] = self.biases[i] - (learning_rate * grad_b[i])
    
    def get_mse_loss(self, data: list[tuple[Matrix, Matrix]]):
        """Calculates the mean squared error loss."""
        total_loss = 0
        n_outputs = self.structure[-1]
        
        for input_vector, expected_output in data:
            output, _ = self.forward_propagate(input_vector)
            error = output - expected_output
            squared_error = error @ error  # Element-wise square
            total_loss += sum(squared_error.get_column(1))  # Sum all squared errors
        
        return total_loss / (len(data) * n_outputs)
    
    def __make_dir(self):
        os.makedirs('cache', exist_ok = True)
    
    def train(self, training_data: list[tuple[Matrix, Matrix]],
              testing_data: list[tuple[Matrix, Matrix]],
              learning_rate: float = 0.1,
              epochs: int = 100,
              plot: bool = False):
        """Trains the neural network based on two different categories of data : training and testing.
        Outputs the MSE loss of each. If "True" is passed to the plot parameter,
        a plot of the evolution  of train losses and test losses will be be displayed and saved 
        in your directory."""
        self.__make_dir()
        train_losses = []
        test_losses = []
        start_date = datetime.now()
        start_time = time.perf_counter()
        for epoch in range(epochs):
            current_lr = learning_rate * (0.95 ** epoch)  # Exponential decay
            for input_vector, expected_output in training_data:
                self.backward_propagate(input_vector, expected_output, current_lr)
            train_loss = self.get_mse_loss(training_data)
            test_loss = self.get_mse_loss(testing_data)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            print(f"Epoch {epoch+1}/{epochs} | Training Loss: {train_loss:.6f} | Testing Loss: {test_loss:.6f}")
        end_date = datetime.now()
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        with open(f'cache/training_info.txt', 'w') as f:
            f.write(f'Epochs: {epochs}.\n')
            f.write(f'Learning rate: {learning_rate}.\n')
            f.write(f'Data size: {len(training_data)+len(testing_data)}. Including:\n \n')
            f.write(f'   Training data size: {len(training_data)}.\n')
            f.write(f'   Testing data size: {len(testing_data)}.\n \n')
            f.write(f'Training start date: {start_date}.\n')
            f.write(f'Training end date: {end_date}.\n')
            f.write(f'Trained in: {time_format(elapsed_time)}.\n')
            f.write(f'Last train loss: {train_losses[-1]}.\n')
            f.write(f'Last test loss: {test_losses[-1]}.')
        if plot:
            plt.figure(figsize=(10, 6))
            plt.plot(train_losses, label='Train Loss')
            plt.plot(test_losses, label='Test Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Mean Squared Error')
            plt.title('Training History')
            plt.legend()
            plt.grid(True)
            plt.savefig(f'cache/training_history.png')
            plt.show()
        return train_losses, test_losses
    
    def save(self, filename:str):
        self.__make_dir()
        """Save model to JSON file"""
        data = {
            'structure': self.structure,
            'hidden_activation': self.hidden_activation_name,
            'output_activation': self.output_activation_name,
            'weights': [W.matrix for W in self.weights],
            'biases': [B.matrix for B in self.biases]
        }
        with open(f'cache/{filename}', 'w') as f:
            json.dump(data, f)

    @classmethod
    def load(cls, filename):
        """Load model from JSON file"""
        with open(filename, 'r') as f:
            data = json.load(f)
        mlp = MultiLayerPerceptron(
            structure=data['structure'],
            hidden_layer_activation_function_label=data['hidden_activation'],
            output_layer_activation_function_label=data['output_activation'],
        )
        mlp.weights = [Matrix(W) for W in data['weights']]
        mlp.biases = [Matrix(B) for B in data['biases']]
        return mlp