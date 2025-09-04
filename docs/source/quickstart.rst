Quickstart
==================

To make sure the pacakge is properly installed,
we can start by performing a simple operation,
such as matrix multiplication:

.. code-block:: python

    from basic_deep_learning import*

    A = Matrix([[-1,1,0,3],[1,3,-1,2],[0,1,0,-1]])
    B = Matrix([[3,-1],[-4,2],[2,1],[-1,0]])

    print(A*B)

The output should be:

.. code-block:: bash

    matrix([
            [-10.0, 3.0],
            [-13.0, 4.0],
            [-3.0, 2.0]
    ])

Let us start by the following example: 
a neural network that picks up on linear patterns
based on the first five terms and predicts the next one.
In your directory, create a python file named 
``training.py``:

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

    train, test = generate_data(2000)

    #Setting up the neural network.

    nn = MultiLayerPerceptron([5, 16, 16, 1], 'ReLU', 'tanh')

    nn.train(train, test, 0.05, decay_rate=0.96, epochs=100, plot=True, plot_epochs_durations=True) #Trainin the model.

    nn.save("nn_test.json") #Saving the model.

Once you run the script, you will see on your terminal:

.. code-block:: bash

    Epoch 1/100 : ███████████████████████████████████████████████████████████████████████████ 100% completed.

    Epoch 1/100 | Training Loss: 0.000524 | Testing Loss: 0.000519 (01 s : 505 ms)

    Epoch 2/100 : ███████████████████████████████████████████████████████████████████████████ 100% completed.

    Epoch 2/100 | Training Loss: 0.000291 | Testing Loss: 0.000270 (01 s : 463 ms)

    Epoch 3/100 : ███████████████████████████████████████████████████████████████████████████ 100% completed.

    Epoch 3/100 | Training Loss: 0.000223 | Testing Loss: 0.000194 (01 s : 449 ms)

    ...

    Epoch 99/100 : ███████████████████████████████████████████████████████████████████████████ 100% completed.

    Epoch 99/100 | Training Loss: 0.000030 | Testing Loss: 0.000032 (01 s : 523 ms)

    Epoch 100/100 : ███████████████████████████████████████████████████████████████████████████ 100% completed.

    Epoch 100/100 | Training Loss: 0.000030 | Testing Loss: 0.000032 (01 s : 270 ms)

In your directory, you should be able to see 
a new directory called ``cache`` in which you will find a 
``nn_test.json`` file, a ``loss_per_epoch.png`` and a ``training_time_per_epoch.png`` image,
and a ``training_info.txt`` file.


.. image:: loss_per_epoch.png

.. image:: training_time_per_epoch.png

.. code-block:: text
    :name: training_info.txt
    :caption: Training informations

    Epochs: 100.
    Learning rate: 0.05.
    Exponential decay rate: 0.96.
    Data size: 2000. Including:
    
       Training data size: 1600.
       Testing data size: 400.
    
    Training start date: 2025-09-04 18:51:33.782316.
    Training end date: 2025-09-04 18:54:33.853151.
    Trained in: 03 m : 00 s : 070 ms.
    Average time per epoch: 01 s : 434 ms/epoch.
    Last training loss: 2.9736125346327896e-05.
    Last testing loss: 3.2130690507638676e-05.

We can now create a new python file named ``loading.py``
in which we will laod the saved model and use it.

.. code-block:: python
    :name: loading.py
    :caption: This file loads the model and uses it.

    from basic_deep_learning import*

    nn = MultiLayerPerceptron.load("cache/nn_test.json")

    def predict_next_term(seq):
        normalized_input = (1/50)*Matrix([seq]).T()
        normalized_output = nn.forward_propagate(normalized_input)[0].get_entry(1,1)
        print(f"The model predicts that the next term of the sequence {seq} is {normalized_output * 50}.")

    predict_next_term([1,2,3,4,5])

Output:

.. code-block:: bash

    The model predicts that the next term of the sequence [1, 2, 3, 4, 5] is 6.095201928726749.

While the model is still not accurate, the error is pretty tolerable given the 
restricted amount of data and the size of the network. Further more, the accuracy could be drastically improved
by increasing the number of hidden layers, the neurons and epochs.
At last, the learning and decay rates can be tweaked for faster convergence as well.

