from random import random
import math

class Polynomial():
    roots = []
    coefficient = []
    order = 0
    derivative = None

    # Given an order, generate random roots or use given roots
    def __init__(self, order=False, roots=False, coefficients=False):
        if order:
            self.order = order
            for i in range(order):
                self.roots.append(complex(random(), random()))
        elif roots:
            self.order = len(roots) 
            self.roots = roots
        elif coefficients:
            self.order = len(coefficients) 
            self.coefficients = coefficients

    def genCoefficients(self):
        coefficients = [1]
        for i in range(self.order):
            coefficients.append((-1)**(i+1) * self.calcSumOfProducts(self.roots, i))
        self.coefficient = coefficients
        print(coefficients)

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
    
    def calcSumOfProducts(self, arr, fix, x=1):
        if fix == 0:
            sum = 0
            for a in arr:
                sum = sum + x*a
            
            return sum
        else:
            sum = 0
            for i in range(len(arr)):
                sum = sum + self.calcSumOfProducts(arr[i+1:], fix-1, x*arr[i])
            return sum

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


poly = Polynomial(roots=[complex(1,0), complex(2,0), complex(3,0), complex(4,0), complex(217,0)])
poly.genCoefficients()
# poly.genDerivative()
# for x in range(-10, 10):
#     for y in range(-10, 10):
#         if findRoot(complex(x,y), poly):
#     else:

