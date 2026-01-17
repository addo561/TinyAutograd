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
    def __init__(self,inputs_n,nonLinearity:str):
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
        if self.nonLinearity=='ReLU':
            y  = y.relu()
        elif self.nonLinearity=='Tanh':
            y = y.tanh()   
        elif self.nonLinearity=='Sigmoid'  :
            y = y.sigmoid()
        return y
    
    def parameters(self):
        #return all parameters ,w is already list so bias should be list 
        return self.w + [self.b]
    
    def __repr__(self):
        return f'{self.nonLinearity}Neuron{self.inputs_n}'
    
class Layer(Module):
    def __init__(self,n_inputs,n_outs,nlinear):
        self.layers = []

        
