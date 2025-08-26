from basic_deep_learning import*
import pytest

tolerance = 1e-10

def test_dot_product():

    assert LinearAlgebraUtils.dot([1,2], [-3,4]) == 5
    assert LinearAlgebraUtils.dot([0,2,0,0], [1,2,3,4]) == 4
    assert LinearAlgebraUtils.dot([1],[-1]) == -1   #Testing cases with different dimensions.

    with pytest.raises(ValueError):
        LinearAlgebraUtils.dot([],[])
    with pytest.raises(ValueError):
        LinearAlgebraUtils.dot([1,2,3],[1,2,3,4])   #Testing error handling.

def test_matrices_initialisation():
    
    M = Matrix([[1,2,3],[4,5,6]])
    assert M.format == (2,3)
    assert M.matrix == [[1,2,3],[4,5,6]]
    assert Matrix([[1]]).format == (1,1)

    N = Matrix([[1,2,3],[4]])
    assert N.format == (2,3)
    assert N.matrix == [[1,2,3],[4,0,0]]    #Test initialisation with "uneven" matrix.

    with pytest.raises(ValueError):
        Matrix([[]])    #Test error handling.

def test_matrix_getters_and_setters():
    
    M = Matrix(([1,2,3,4],[6,7,8,9,10],[11,12]))
    assert M.get_entry(3,3) == 0
    M.set_entry(-1,3,3)
    assert M.get_entry(3,3) == -1

    with pytest.raises(ValueError):
        M.get_entry(0,1)
    with pytest.raises(ValueError):
        M.get_entry(4,3)

    assert M.get_row(3) == [11,12,-1,0,0]
    assert M.get_column(2) == [2,7,12]

    assert Matrix.zero(2,3).matrix == [[0,0,0],[0,0,0]]

    with pytest.raises(ValueError):
        Matrix.zero(0,2)

def test_matrix_operations():
    
    I = Matrix([[1,0,0],[0,1,0],[0,0,1]])
    assert I.T() == I
    
    M = Matrix([[1,2,3],[5,6,7]])
    assert M.T().format == (3,2)
    assert M.T().matrix == [[1,5],[2,6],[3,7]]

    A = Matrix([[1,2],[3,4]])
    B = Matrix([[-1,3],[-4,2]])
    k = 2
    S = Matrix([[0,5],[-1,6]])
    D = Matrix([[2, -1],[7, 2]])
    EWP = Matrix([[-1, 6],[-12,8]])
    P = Matrix([[-9,7],[-19,17]])
    SCALED_A = Matrix([[2,4],[6,8]])

    assert A+B == S
    assert A-B == D
    assert A@B == EWP
    assert A*B == P
    assert k*A == SCALED_A

    with pytest.raises(ValueError):
        A + M
        A - M
        A @ M
        M * A

    assert A * M == Matrix([[11, 14, 17], [23, 30, 37]])

def test_random_matrix():
    M = Matrix.randomize(3,2,-2,2)
    assert M.format == (3,2)
    for i in range(1,4):
        for j in range(1,3):
            assert -2 <= M.get_entry(i,j) <= 2

def test_activation_functions():
    
    #sigmoid_test
    assert abs(ActivationFunctionsRegistry.sigmoid(1) - 0.73105857863) < tolerance
    assert abs(ActivationFunctionsRegistry.sigmoid([-1,0,1])[0] - 0.26894142137) < tolerance
    M = Matrix([[1,0],[0,-1]])
    S = Matrix([[0.73105857863, 0.5],[0.5, 0.26894142137]])
    D = ActivationFunctionsRegistry.sigmoid(M) - S
    for i in range(1,3):
        for j in range(1,3):
            assert abs(D.get_entry(i,j)) < tolerance

    Sp = Matrix([[0.196611933241,0.25],[0.25,0.196611933241]])
    Dp = ActivationFunctionsRegistry.Activations["sigmoid"][1](M) - Sp
    for i in range(1,3):
        for j in range(1,3):
            assert abs(Dp.get_entry(i,j)) < tolerance

    with pytest.raises(ValueError):
        ActivationFunctionsRegistry.softmax(Matrix([[-10],[2],[3],[1],[5,2]]))

    B = Matrix([[-10],[2],[3],[1],[5]])
    P = ActivationFunctionsRegistry.softmax(B)
    s = 0
    for j in range(1,6):
        s += P.get_entry(j,1)
        assert 0 <= P.get_entry(j, 1) <= 1
    assert s == 1

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
