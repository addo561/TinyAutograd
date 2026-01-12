import math

class Value:
    def __init__(self,data):
        self.data = data
        self.grad = 0
    def __repr__(self):
        return f'Value(data=[{self.data}],grad=[{self.grad}])'
        
    def __add__(self,other):
        other = other if  isinstance(other,Value) else Value(other)
        out = Value(self.data + other.data)
        return out 
        
    def __mul__(self,other):
        other = other if  isinstance(other,Value) else Value(other)
        out = Value(self.data * other.data)
        return out
        
    def __pow__(self,other):
        out = Value(self.data ** other)
        return out 
        
    def __sub__(self,other):
        return self + (-other)
        
    def __rmul__(self,other):
        return self * other
        
    def __rsub__(self,other):
        return other + (-self)
        
    def __neg__(self):
        return self * -1
        
    def __truediv__(self,other):
        return self * other**-1
        
    def __rtruediv__(self,other):
        return other * self**-1   
        
    def backward(self):
        pass
        
    def __radd__(self,other):
        return  self + other
        
    def relu(self):
        out = Value(0 if self.data < 0 else self.data)
        return out
        
    def tanh(self):
        out = Value((math.exp(self.data) - math.exp(-self.data)) /(math.exp(self.data) + math.exp(-self.data)) )
        return  out 
    def sigmoid(self):
        return Value(1 / (1 + math.exp(-self.data)))
        