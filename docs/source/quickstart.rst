Quickstart
==================

To make sure the pacakge is properly install,
we can start by performing a simple operation,
such as matrix multiplication:

.. code-block:: python

    from basic_deep_learning import*

    A = Matrix([[-1,1,0,3],[1,3,-1,2],[0,1,0,-1]])
    B = Matrix([[3,-1],[-4,2],[2,1],[-1,0]])

    print(A*B)

The output should be:

.. code-block:: bash

    matrix([[-10,3],[-13,4],[-3,2]])

Let us start by a classic example: a neural network
that picks up on XOR gates patterns. 
Even though a computer can deterministically 
compute the output of an XOR operation, 
it still serves as a good example to begin with.

.. code-block:: python
    :name: training.py
    :caption: This file initialises the neural network, trains it and saves it.

    from  basic_deep_learning import*
    import random


    #A function that will generate various data of linear patterns.
    def generate_data(n=1000):
        data = []

        for i in range(n):
            start = random.randint(-10,10)
            step = random.randint(-5,5)
            seq = [start + j*step for j in range(5)]
            next_term = start + 5*step

            # Normalize by 50 to prevent floating-point errors.
            max_in_seq = max(seq)
            normalized_seq = [x / 50 for x in seq]
            normalized_next = next_term / 50

            input_matrix = Matrix([normalized_seq]).T()
            expected_output_matrix = Matrix([[normalized_next]])

            data.append((input_matrix, expected_output_matrix))
        
        random.shuffle(data)
        split_index = int(0.8 * len(data))
        return data[:split_index], data[split_index:]

    train, test = generate_data(500)

    #Setting up the neural network.

    nn = MultiLayerPerceptron([5,16,16,1], 'tanh', 'ReLU')

    nn.train(train, test, 0.05, 100, True) #Trainin the model.

    nn.save("nn_test.json") #Saving the model.
