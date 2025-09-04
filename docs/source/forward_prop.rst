Forward Propagation
-------------------

Definition:

.. code-block:: python

    forward_propagate(input_vector: Matrix): -> tuple[Matrix, tuple[list[Matrix], list[Matrix]]]

If ``input_matrix`` is not a column vector, a ``ValueError`` will be raised.

Let :math:`L` be the number of layers, :math:`\left(W^{(1)},\cdots,W^{(L-1)}\right)` be the 
:math:`L-1` weight matrices and :math:`\left(B^{(1)},\cdots,B^{(L-1)}\right)` be the 
:math:`L-1` bias matrices. We iteratively compute the output of the MLP 
based on the input vector :math:`A^{(0)}=X` via the recursive formula:

.. math::

    \forall i \in [1, L], \quad A^{(i)} = f\left(W^{(i)}\cdot A^{(i-1)} + B^{(i)}\right),

where :math:`f` is the hidden layers activation function if :math:`i < L` and 
is the output layer activation function if :math:`i = L`.
The output is hence the last column matrix :math:`A^{(L)}`.

The method returns a tuple ``(output_vector, (activations, pre_activations))``.

``output_vector`` is self explanatory: the :math:`A^{(L)}` vector as a ``Matrix`` instance.

``activations``: the list of the :math:`A^{(i)}` column vectors.

``pre_activations``: the list of the :math:`A^{(i)}` vectors before evaluating the activation function,
meaning the :math:`Z^{(i)}` column vectors (:math:`1\leq i \leq L`) where:

.. math::

    \forall i \in [1, L], \quad Z^{(i)} = W^{(i)}\cdot A^{(i-1)} + B^{(i)}\right,

