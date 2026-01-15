import math

class Value:
    def __init__(self, data, children=()):
        # The actual number (e.g., 3.0)
        self.data = data
        # The derivative of the final loss with respect to this value.
        # It starts at 0 because at first, this node has no effect on anything.
        self.grad = 0
        
        # This is the function that knows how to calculate the gradient for this specific step.
        # By default (for input nodes), it does nothing.
        self._backward = lambda: None
        
        # We save the 'children' (the nodes that created this one) 
        # so we can rebuild the graph later.
        self._prev = set(children)

    def __repr__(self):
        return f'Value(data={self.data}, grad={self.grad})'


    def __add__(self, other):
        # If 'other' is just a number (like 2), turn it into a Value object first
        other = other if isinstance(other, Value) else Value(other)
        
        # Create the new node. We must list (self, other) as children to link the graph.
        out = Value(self.data + other.data, (self, other))

        # Define how to propagate the gradient backwards for addition.
        # Addition just passes the gradient equally to both parents.
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
            
        # IMPORTANT: Store this function on the RESULT node (out), not the input (self).
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other))

        # For multiplication, the gradient of one side depends on the *data* of the other side.
        # (Chain Rule: d(xy)/dx = y)
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
            
        out._backward = _backward
        return out

    def __pow__(self, other):
        # Note: 'other' is expected to be a simple number (int/float), not a Value object here.
        out = Value(self.data ** other, (self,))

        # Power Rule: d(x^n)/dx = n * x^(n-1)
        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
            
        out._backward = _backward
        return out

    # --- Activation Functions (Non-linearities) ---
    
    def relu(self):
        # If data < 0, output is 0. Otherwise, pass it through.
        # We MUST pass (self,) as a child or the graph breaks here!
        out = Value(0 if self.data < 0 else self.data, (self,))

        def _backward():
            # If the value was negative (and reset to 0), no gradient passes through.
            # If it was positive, the gradient passes through unchanged (x1).
            self.grad += (out.data > 0) * out.grad
            
        out._backward = _backward
        return out

    def tanh(self):
        # The math formula for tanh
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self,))
        
        def _backward():
            # Derivative of tanh is (1 - tanh^2)
            self.grad += (1 - t**2) * out.grad
            
        out._backward = _backward
        return out

    def sigmoid(self):
        x = self.data
        # Sigmoid formula: 1 / (1 + e^-x)
        s = 1 / (1 + math.exp(-x))
        out = Value(s, (self,))
        
        def _backward():
            # Derivative of sigmoid is s * (1 - s)
            self.grad += (s * (1 - s)) * out.grad
            
        out._backward = _backward
        return out
    
    def __neg__(self):
        # Allows you to type "-a"
        return self * -1

    def __sub__(self, other):
        # Allows you to type "a - b" (it becomes a + (-b))
        return self + (-other)

    def __radd__(self, other):
        # Allows you to type "2 + a" (Python flips it to a + 2)
        return self + other

    def __rmul__(self, other):
        # Allows you to type "2 * a"
        return self * other
        
    def __truediv__(self, other):
        # Allows "a / b" (it becomes a * b^-1)
        return self * other**-1

    # The Engine (Backpropagation) ---

    def backward(self):
        # 1. Topological Sort
        # We need to lay out the graph flat so we visit parents only after children.
        topo = []
        visited = set()
        
        def build_topo(node):
            if node in visited:
                return
            visited.add(node)
            # Recursively visit all children first
            for child in node._prev:
                build_topo(child)
            # Once children are processed, add myself to the list
            topo.append(node)
        
        build_topo(self)
        
        # 2. Go backwards
        # Start the chain reaction by setting the final node's gradient to 1.0
        self.grad = 1.0
        
        # Go through the list in reverse order (Output -> Input)
        for node in reversed(topo):
            node._backward()