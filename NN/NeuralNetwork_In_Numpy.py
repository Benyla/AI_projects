import numpy as np 
from typing import Sequence
from functools import reduce
from math import exp, log

## << Engine >> ##
class Tensor:
    def __init__(self, data: float, grad_fn=lambda: []):
        self.data = data
        self.grad_fn = grad_fn
        self.grad = 0.0

    def __add__(self: 'Tensor', other: 'Tensor') -> 'Tensor': return Tensor(self.data + other.data, lambda: [(self, 1.0), (other, 1.0)])
    def __mul__(self: 'Tensor', other: 'Tensor') -> 'Tensor': return Tensor(self.data * other.data, lambda: [(self, other.data), (other, self.data)])
    def __pow__(self, power): assert type(power) in {float, int}; return Tensor(self.data ** power, lambda: [(self, power * self.data ** (power - 1))]) # Supports only int and float
    def __neg__(self: 'Tensor') -> 'Tensor': return Tensor(-1.0) * self
    def __sub__(self: 'Tensor', other: 'Tensor') -> 'Tensor': return self + (-other)
    def __truedidata__(self: 'Tensor', other: 'Tensor') -> 'Tensor': return self * other ** -1 # Division
    def __repr__(self) :return "Tensor(data=%.4f, grad=%.4f)" % (self.data, self.grad)
    def exp(self): return Tensor(exp(self.data), lambda: [(self, exp(self.data))])
    def log(self): return Tensor(log(self.data), lambda: [(self, self.data ** -1)])
    
    ## << Activation Functions >> ##

    def relu(self): return Tensor(self.data if self.data > 0.0 else 0.0, lambda: [(self, 1.0 if self.data > 0.0 else 0.0)])
    def identity(self): return Tensor(self.data, lambda: [(self, 1.0)])
    def tanh(self): tanh_val = np.tanh(self.data); return Tensor(tanh_val, lambda: [(self, 1 - tanh_val ** 2)])
    def sigmoid(self): sigmoid_val = 1 / (1 + np.exp(-self.data)); return Tensor(sigmoid_val, lambda: [(self, sigmoid_val * (1 - sigmoid_val))])

    ## << Backward >> ##

    def backward(self, bp=1.0): self.grad += bp; [input.backward(grad * bp) for input, grad in self.grad_fn()]


## << nn >> ##

class Initializer: # Modify to include an option for normal initializer to allow for Glorot and He initialization
    def init_weights(self, n_in, n_out): return [[Tensor(np.random.uniform(-1, 1)) for _ in range(n_out)] for _ in range(n_in)]
    def init_bias(self, n_out): return [Tensor(np.random.uniform(-1, 1)) for _ in range(n_out)]

class DenseLayer:
    def __init__(self, n_in: int, n_out: int, act_fn, initializer = Initializer()):
        self.weights = initializer.init_weights(n_in, n_out)
        self.bias = initializer.init_bias(n_out)
        self.act_fn = act_fn

    def __repr__(self): return 'Weights: ' + repr(self.weights) + ' Biases: ' + repr(self.bias) # Helper func to print parms

    def forward(self, input: Sequence[Tensor]) -> Sequence[Tensor]: # Should be called for each layer
        out = []
        for j in range(len(self.bias)):  # Number of neuron in current layer
            node = self.bias[j]  # Add the bias j'th neuron to the final result for the j'th neuron 
            for i in range(len(input)):  # Iterate over number of input neurons
                node += input[i] * self.weights[i][j]  # Accumulate weighted sum (Tensor * Tensor)
            out.append(self.act_fn(node))  # Apply activation function
        return out
    
    def parameters(self) -> Sequence[Tensor]: # flattens the weight matrix for a given layer 
        flat_weights = [p for row in self.weights for p in row]; return flat_weights + self.bias
    
## << Loss Functions >> ##

class Loss:
    def __call__(self, ycs: Sequence[Tensor], yps: Sequence[Tensor]) -> Tensor: return self.loss_func(ycs, yps) # to insure instances of Loss can be called as a function

class MSELoss(Loss):
    def __init__(self): 
        self.loss_func = lambda ycs, yps: sum(((yc - yp)**2 for yc, yp in zip(ycs, yps)), Tensor(0.0)) # will fail on list of lists - see type hints
                                                                                                        # if network has more than 1 output neuron we need to modify the loss func

## << Complete forward_pass and helper funcs >> ##

def forward_pass(NN, x: Sequence[Tensor]) -> Sequence[Sequence[Tensor]]: # This function returns a complete forward pass for observations, x, through the entire network
    # will still return list of lists eventhough last layer has 1 output neuron - needs to flattened before passed to loss func 
    return [reduce(lambda out, layer: layer.forward(out), NN, single_input) for single_input in x] # reduce(function, iterable, initializer), where NN=iter, single_input=init

def parameters(network) -> Sequence[Sequence[Tensor]]: # This function returns a list of lists, where each list contains all the parameters for a given layer
    return [p for layer in network for p in layer.parameters()]

def update_parameters(params, learning_rate=0.01):
  for p in params:
    p.data -= learning_rate*p.grad

def zero_gradients(params):
  for p in params:
    p.grad = 0.0


## << Building the network >> ##

# NN = [
#     DenseLayer(1, 16, lambda x: x.relu()),  # First layer
#     DenseLayer(16, 1, lambda x: x.identity()) # Second layer
# ]

# mse_loss = MSELoss() # define loss function

# build training loop....

