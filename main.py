from random import random
import math

def truncate(number, digits) -> float:
    nbDecimals = len(str(number).split('.')[1]) 
    if nbDecimals <= digits:
        return number
    stepper = 10.0 ** digits
    return math.trunc(stepper * number) / stepper

class Polynomial():
    coefficients = []
    order = 0
    derivative = None

    # Given an order, generate random coefficients
    def __init__(self, order):
        self.order = order
        for i in range(order):
            self.coefficients.append(random())

    # Coefficients given
    def __init__(self, coefficients):
        self.order = len(coefficients) 
        self.coefficients = coefficients

    def genDerivative(self):
        derivCoes = []
        for i in range(len(self.coefficients)-1, 0, -1):
            derivCoes.append(self.coefficients[self.order-i-1] * i)
        self.derivative = Polynomial(derivCoes)

    def calc(self, x):
        result = self.coefficients[0]
        for c in range(1, self.order):
            result = result * x + self.coefficients[c]
        return result
    
    def calcDerivative(self, x):
        if self.derivative == None:
            raise NameError("Derivative not generated")
        return self.derivative.calc(x)

def newtonRaphson(x, poly):
    return x - poly.calc(x)/poly.calcDerivative(x)

def findRoot(x0, poly):
    iterCount = 0
    result = x0
    while not math.isclose(poly.calc(result), 0, rel_tol=1e-10) and iterCount < 100:
        result = newtonRaphson(result, poly) 
        iterCount = iterCount + 1
    return result
    
poly = Polynomial([1,1,1,1,1,1])
poly.genDerivative()
print(findRoot(1, poly))
