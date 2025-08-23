from basic_deep_learning.functionality.activations_registry import ActivationFunctionsRegistry
from basic_deep_learning.functionality.matrix import Matrix
from basic_deep_learning.miscellaneous.decorators import extend_to_matrices
from basic_deep_learning.miscellaneous.linear_algebra import LinearAlgebraUtils
from basic_deep_learning.models.mlp import MultiLayerPerceptron

import pytest
import os


def test_matrix_operations():
    """Test basic matrix operations."""
    # Test initialization
    m1 = Matrix([[1, 2], [3, 4]])
    assert m1.format == (2, 2)
    assert m1.matrix == [[1, 2], [3, 4]]
    
    # Test uneven rows
    m2 = Matrix([[1], [2, 3]])
    assert m2.format == (2, 2)
    assert m2.matrix == [[1, 0], [2, 3]]
    
    # Test addition
    m3 = Matrix([[1, 2], [3, 4]])
    m4 = Matrix([[5, 6], [7, 8]])
    result = m3 + m4
    assert result.matrix == [[6, 8], [10, 12]]
    
    # Test multiplication
    m5 = Matrix([[1, 2, 3], [4, 5, 6]])
    m6 = Matrix([[7, 8], [9, 10], [11, 12]])
    result = m5 * m6
    assert result.format == (2, 2)
    assert result.matrix == [[58, 64], [139, 154]]
    
    # Test transpose
    m7 = Matrix([[1, 2], [3, 4], [5, 6]])
    transposed = m7.T()
    assert transposed.format == (2, 3)
    assert transposed.matrix == [[1, 3, 5], [2, 4, 6]]

def test_activation_functions():
    """Test activation functions and their derivatives."""
    # Test sigmoid
    assert abs(ActivationFunctionsRegistry.sigmoid(0) - 0.5) < 1e-10
    assert abs(ActivationFunctionsRegistry.sigmoid_prime(0) - 0.25) < 1e-10
    
    # Test ReLU
    assert ActivationFunctionsRegistry.ReLU(1) == 1
    assert ActivationFunctionsRegistry.ReLU(-1) == 0
    assert ActivationFunctionsRegistry.ReLU_prime(1) == 1
    assert ActivationFunctionsRegistry.ReLU_prime(-1) == 0
    
    # Test softmax
    test_vector = Matrix([[1], [2], [3]])
    softmax_result = ActivationFunctionsRegistry.softmax(test_vector)
    total = sum(softmax_result.get_entry(i, 1) for i in range(1, 4))
    assert abs(total - 1.0) < 1e-10
    
    # Test with large values (numerical stability)
    large_vector = Matrix([[1000], [1001], [1002]])
    softmax_result = ActivationFunctionsRegistry.softmax(large_vector)
    total = sum(softmax_result.get_entry(i, 1) for i in range(1, 4))
    assert abs(total - 1.0) < 1e-10

def test_mlp_operations():
    """Test MLP initialization and basic operations."""
    # Test initialization
    mlp = MultiLayerPerceptron([2, 3, 1], 'sigmoid', 'sigmoid')
    assert mlp.structure == [2, 3, 1]
    assert len(mlp.weights) == 2
    assert len(mlp.biases) == 2
    
    # Test forward propagation
    input_vec = Matrix([[0.5], [0.5]])
    output, (activations, pre_activations) = mlp.forward_propagate(input_vec)
    assert output.format == (1, 1)
    assert len(activations) == 3
    assert len(pre_activations) == 2
    
    # Test backpropagation
    expected_output = Matrix([[1.0]])
    initial_loss = mlp.get_mse_loss([(input_vec, expected_output)])
    mlp.backward_propagate(input_vec, expected_output, learning_rate=0.1)
    final_loss = mlp.get_mse_loss([(input_vec, expected_output)])
    assert final_loss < initial_loss  # Loss should decrease

def test_mlp_training():
    """Test MLP training on a simple problem."""
    # Create simple XOR-like dataset
    train_data = [
        (Matrix([[0], [0]]), Matrix([[0]])),
        (Matrix([[0], [1]]), Matrix([[1]])),
        (Matrix([[1], [0]]), Matrix([[1]])),
    ]
    test_data = [
        (Matrix([[1], [1]]), Matrix([[0]]))
    ]
    
    # Create and train MLP
    mlp = MultiLayerPerceptron([2, 4, 1], 'sigmoid', 'sigmoid')
    train_losses, test_losses = mlp.train(
        training_data=train_data,
        testing_data=test_data,
        learning_rate=0.05,
        epochs=100,
        plot=False
    )
    
    # Check that losses are decreasing
    assert train_losses[-1] < train_losses[0]
    assert test_losses[-1] < test_losses[0]

def test_linear_algebra():
    """Test linear algebra utilities."""
    # Test dot product
    assert LinearAlgebraUtils.dot([1, 2, 3], [4, 5, 6]) == 32
    assert LinearAlgebraUtils.dot([0, 0], [0, 0]) == 0
    assert LinearAlgebraUtils.dot([-1, 2], [3, -4]) == -11
    
    # Test error cases
    with pytest.raises(ValueError):
        LinearAlgebraUtils.dot([], [1, 2])
    with pytest.raises(ValueError):
        LinearAlgebraUtils.dot([1, 2], [1])

def test_matrix_serialization():
    """Test matrix serialization for MLP save/load."""
    # Create and train a simple MLP
    mlp = MultiLayerPerceptron([2, 3, 1], 'sigmoid', 'sigmoid')
    train_data = [(Matrix([[0.5], [0.5]]), Matrix([[1.0]]))]
    mlp.train(train_data, train_data, epochs=1, plot=False)
    
    # Save and load
    mlp.save('test_model.json')
    loaded_mlp = MultiLayerPerceptron.load('test_model.json')
    
    # Check if loaded model matches original
    assert loaded_mlp.structure == mlp.structure
    assert loaded_mlp.hidden_activation_name == mlp.hidden_activation_name
    assert loaded_mlp.output_activation_name == mlp.output_activation_name
    
    # Clean up
    import os
    if os.path.exists('test_model.json'):
        os.remove('test_model.json')

if __name__ == "__main__":
    pytest.main([__file__, "-v"])