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

    # Given an order, generate random coefficients or use given coefficients
    def __init__(self, order=False, coefficients=False):
        if order:
            self.order = order
            for i in range(order):
                self.coefficients.append(random())
        elif coefficients:
            self.order = len(coefficients) 
            self.coefficients = coefficients

    def genDerivative(self):
        derivCoes = []
        for i in range(len(self.coefficients)-1, 0, -1):
            derivCoes.append(self.coefficients[self.order-i-1] * i)
        self.derivative = Polynomial(coefficients=derivCoes)

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
    if poly.calcDerivative(x) == 0:
        return x, True # Won't find root
    return x - poly.calc(x)/poly.calcDerivative(x), False

def findRoot(x0, poly):
    iterCount = 0
    result = x0
    diverges = False
    while not math.isclose(poly.calc(result.real), 0, abs_tol=1e-2) and not math.isclose(poly.calc(result.imag), 0, abs_tol=1e-2) and iterCount < 100 and not diverges:
        result, diverges = newtonRaphson(result, poly) 
        iterCount = iterCount + 1
    if iterCount == 100 or diverges:
        return False # Root not found
    else:
        return True # Root found


poly = Polynomial(order=5)
poly.genDerivative()
# for x in range(-10, 10):
#     for y in range(-10, 10):
#         if findRoot(complex(x,y), poly):
#     else:

