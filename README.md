# TinyAutograd

A minimal automatic differentiation engine built from scratch to understand 
how backpropagation and neural networks work under the hood.

## Automatic Differentiation

`Engine` - scalar-valued autograd engine with a dynamic computation graph. 
Tracks operations and runs reverse-mode backpropagation via `.backward()`.

## Custom Tensor Engine

`Tensor.ipynb` - extended the engine to support NumPy-backed tensors, 
built a custom neural network on top, and trained it on MNIST.

### MNIST Results — 75% Accuracy

<!-- Add sample predictions image here -->
<!-- ![MNIST Samples](assets/mnist_samples.png) --><img width="629" height="443" alt="output" src="https://github.com/user-attachments/assets/c00c81b2-14f3-4005-b9db-f8d4b7201e8c" />
