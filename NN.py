### PYTORCH-LIKE ABSTRACTIONS
from engine import Value
import random
class Module:
    def parameters(self):
        #list of all Value objects  that need updating
        return []
    def zero_grad(self):
        #sets  all parameters grads to zero
        for p in self.parameters:
            p.grad = 0


class Neuron(Module):
    def __init__(self,inputs_n,nonLinearity=True):
        #weights  the same size as input 
        self.w = [Value(random.uniform(0,1)) for _ in range(inputs_n)] 
        #bias is just added 
        self.b = Value(1)
        # non- linearity  whether relu,tanh
        self.nonLinearity  = nonLinearity
        self.inputs_n  = inputs_n

    def __call__(self,x):
        # Does the y = wx + b
        y = sum(( w * x for w,x in zip(self.w,x)),self.b)
        if self.nonLinearity:
            y  = y.relu()
        else:
            y = y.tanh()   
        return y
    
    def parameters(self):
        #return all parameters ,w is already list so bias should be list 
        return self.w + [self.b]
    
    def __repr__(self):
        return f"{'ReLU' if self.nonLinearity else 'Linear'}Neuron({len(self.w)})"
    
class Layer(Module):
    def __init__(self,n_inputs,n_outs,**kwargs):
        self.neurons = [Neuron(n_inputs,**kwargs) for _ in range(n_outs)]

    def __call__(self,x):
        o = [n(x) for n in self.neurons]
        return o[0] if len(o) == 1 else o
    
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]
    
    def __repr__(self):
        return f'Layer of {', '.join(str(n) for n in self.neurons)}'


class MLP(Module):

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1],nonLinearity=i!=len(nouts)-1) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"