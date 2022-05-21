from random import random

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


poly = Polynomial([1,2,3])
poly.genDerivative()
print(poly.calc(1))
print(poly.calcDerivative(1))
