
from typing import Sequence, Tuple
import numpy as np
import sympy
import matplotlib.pyplot as plt
import cirq

"""Define problem parameters and get a set of GridQubits."""
# Set the dimensions of the grid.
# n_cols = 3
n_rows = 10

# Set the value of the external magnetic field at each site.
# h = 0.5 * np.ones((n_rows, n_cols))
h = 0.5 * np.ones(n_rows)

# Arranging the qubits in a list-of-lists like this makes them easy to refer to later.
# qubits = [[cirq.GridQubit(i, j) for j in range(n_cols)] for i in range(n_rows)]
qubits = [cirq.LineQubit(i) for i in range(n_rows)]

def gamma_layer(gamma_value: float, h: np.ndarray) -> Sequence[cirq.Operation]:
    """Generator for U(gamma, C) layer of QAOA

    Args:
        gamma: Float variational parameter for the circuit
        h: Array of floats of external magnetic field values
    """
    for i in range(n_rows):
        # for j in range(n_cols):
            if i < n_rows - 1:
                # yield cirq.ZZ(qubits[i][j], qubits[i + 1][j]) ** gamma_value
                yield cirq.ZZ(qubits[i], qubits[i + 1]) ** gamma_value
            # if j < n_cols - 1:
                # yield cirq.ZZ(qubits[i][j], qubits[i][j + 1]) ** gamma_value
            # yield cirq.Z(qubits[i][j]) ** (gamma_value * h[i, j])
            yield cirq.Z(qubits[i]) ** (gamma_value * h[i])
            
            
def beta_layer(beta_value: float) -> Sequence[cirq.Operation]:
    """Generator for U(beta, B) layer (mixing layer) of QAOA"""
    for row in qubits:
        # for qubit in row:
            # yield cirq.X(qubit) ** beta_value
        yield cirq.X(row) ** beta_value
            


"""Create the QAOA circuit."""
# Use sympy.Symbols for the 𝛾 and β parameters.
gamma = sympy.Symbol("𝛄")
beta = sympy.Symbol("β")

# Start in the H|0> state.
qaoa = cirq.Circuit(cirq.H.on_each(qubits))

# Implement the U(gamma, C) operator.
qaoa.append(gamma_layer(gamma, h))

# Implement the U(beta, B) operator.
qaoa.append(beta_layer(beta), strategy=cirq.InsertStrategy.NEW_THEN_INLINE)

# Display the QAOA circuit.
print(qaoa)