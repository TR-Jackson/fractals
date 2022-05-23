from random import random
import math
import matplotlib.pyplot as plt
import numpy as np

class Polynomial():
    roots = []
    coefficients = []
    order = 0
    derivative = None

    # Given an order, generate random roots or use given roots
    def __init__(self, order=False, roots=False, coefficients=False):
        if order:
            # bad, needs to take into account complex root pairs
            self.order = order
            for i in range(order):
                self.roots.append(complex(random(), random()))
        elif roots:
            self.order = len(roots) 
            self.roots = roots
        elif coefficients:
            self.order = len(coefficients)-1
            self.coefficients = coefficients

    def genCoefficients(self):
        coefficients = [1.0]
        for i in range(self.order):
            coefficients.append(((-1)**(i+1) * self.calcSumOfProducts(self.roots, i)).real)
        self.coefficients = coefficients

    def genDerivative(self):
        derivCoes = []
        for i in range(len(self.coefficients)-1):
            derivCoes.append((len(self.coefficients)-1-i)*self.coefficients[i])
        self.derivative = Polynomial(coefficients=derivCoes)

    def calc(self, x):
        result = self.coefficients[0]
        for c in range(1, self.order+1):
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
    fdash = poly.calcDerivative(x)
    f = poly.calc(x)
    if fdash == 0:
        return x, True # Won't find root
    return x - f/fdash, False

def findRoot(x0, poly):
    iterCount = 0
    maxIter = 100
    result = poly.calc(x0)
    root = x0
    diverges = False
    while not (math.isclose(result.real, 0, abs_tol=1e-2) and math.isclose(result.imag, 0, abs_tol=1e-2)) and iterCount < maxIter and not diverges:
        root, diverges = newtonRaphson(root, poly)
        result = poly.calc(root)
        iterCount = iterCount + 1
    if iterCount == maxIter or diverges:
        return False # Root not found
    else:
        return root # Root found

# poly = Polynomial(roots=[complex(0,1), complex(0,-1), complex(1,0), complex(1,0), complex(1/2,math.sqrt(3)/2), complex(-1/2,math.sqrt(3)/2), complex(1/2,-math.sqrt(3)/2), complex(-1/2,-math.sqrt(3)/2), complex(math.sqrt(3)/2,1), complex(math.sqrt(3)/2,-1), complex(-math.sqrt(3)/2,1), complex(math.sqrt(3)/2,-1)])
poly = Polynomial(roots=[complex(0,1), complex(0,-1), complex(1,0), complex(1,0), complex(1,1), complex(1,-1)])
poly.genCoefficients()
print(poly.roots)
print(poly.coefficients)
poly.genDerivative()
size = 10
step = 0.01
xCount = 0
points = np.ndarray(shape=(int((size/step)*2),int((size/step)*2)))
for x in np.arange(-size, size, step):
    yCount = 0
    for y in np.arange(-size, size, step):
        root = findRoot(complex(x,y), poly)
        if root:
            for r in range(len(poly.roots)):
                if math.isclose(root.real, poly.roots[r].real, abs_tol=1e-1) and math.isclose(root.imag, poly.roots[r].imag, abs_tol=1e-1):
                    points[xCount][yCount] = r
        yCount = yCount + 1
    xCount = xCount + 1

plt.matshow(points)
plt.show()