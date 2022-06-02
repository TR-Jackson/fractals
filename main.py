from random import random
import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap
import multiprocess as mp
import time

class Polynomial():
    roots = []
    coefficients = []
    order = 0
    derivative = None

    def __init__(self, order=False, roots=False, coefficients=False):
        if order:
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
    fPrime = poly.calcDerivative(x)
    f = poly.calc(x)
    if fPrime == 0:
        return x, True # Won't find root
    return x - f/fPrime, False

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



if __name__ == "__main__":
    isLoad = input("Load fractal? y/n;   ")
    if isLoad == "y":
        points = np.loadtxt(input("Enter fractal name;   ") + ".txt")
        xRes = points.shape[1]
        yRes = points.shape[0]
    else:
        if input("Enter roots custom roots? y/n;    ") == "y":
            done = False
            roots = []
            while not done:
                real = float(input("Enter the real part of the root;   "))
                imag = float(input("Enter the imaginary part of the root;   "))
                roots.append(complex(real,imag))
                if imag != 0:
                    roots.append(complex(real,-imag))
                if input("Add another root? y/n;    ") != "y":
                    done = True
        else:
            # roots = [complex(0,1), complex(0,-1), complex(1,0), complex(math.sqrt(2),math.sqrt(3)), complex(math.sqrt(2),-math.sqrt(3)), complex(1,math.sqrt(3)), complex(1,-math.sqrt(3))]
            roots = [complex(1,1), complex(1,-1), complex(0,1), complex(0,-1), complex(0,0)]

        xStart = float(input("Enter start x bound;    "))
        xEnd = float(input("Enter end x bound;    "))
        yStart = float(input("Enter start y bound;    "))
        yEnd = float(input("Enter end y bound;    "))

        print("Standard resolutions: 640x480, 1280x720, 1920x1080, 2560x1440, 3840x2160, 7680x4320")
        xRes = int(input("Enter x resolution;   "))
        yRes = int(input("Enter y resolution;   "))

        xStepSize = abs((xEnd-xStart)/xRes)
        yStepSize = abs((yEnd-yStart)/yRes)

        poly = Polynomial(roots=roots)
        poly.genCoefficients()
        print(poly.roots)
        print(poly.coefficients)
        poly.genDerivative()

        points = np.ndarray(shape=(yRes, xRes))

        if input("Use multiprocessing? y/n;   ") == "y":
            start = time.perf_counter()
            # More processes with fewer tasks each
            # def f(x, xIndex, yStart, yEnd, yStepSize, yRes, poly):
            #     res = []
            #     for y in np.arange(yStart, yEnd, yStepSize):
            #         yIndex = int(yRes-1-(y-yStart)/yStepSize)
            #         root = findRoot(complex(x,y), poly)
            #         if root:
            #             for r in range(len(poly.roots)):
            #                 if math.isclose(root.real, poly.roots[r].real, abs_tol=1e-1) and math.isclose(root.imag, poly.roots[r].imag, abs_tol=1e-1):
            #                     res.append([xIndex, yIndex, r])
            #     return res

            # results = []

            # with mp.Pool(mp.cpu_count()) as pool:
            #     for x in np.arange(xStart, xEnd, xStepSize):
            #         xIndex = int((x-xStart)/xStepSize)
            #         if (xIndex+1 == xRes): print("Generation complete")
            #         results.append(pool.apply_async(f, (x,xIndex, yStart, yEnd, yStepSize, yRes, poly,)))

            def f(y, yIndex, xStart, xEnd, xStepSize, poly):
                res = []
                for x in np.arange(xStart, xEnd, xStepSize):
                    xIndex = int((x-xStart)/xStepSize)
                    root = findRoot(complex(x,y), poly)
                    if root:
                        for r in range(len(poly.roots)):
                            if math.isclose(root.real, poly.roots[r].real, abs_tol=1e-1) and math.isclose(root.imag, poly.roots[r].imag, abs_tol=1e-1):
                                res.append([xIndex, yIndex, r])
                return res

            results = []

            with mp.Pool(mp.cpu_count()) as pool:
                for y in np.arange(yStart, yEnd, yStepSize):
                    yIndex = int(yRes-1-(y-yStart)/yStepSize)
                    if (yIndex+1 == yRes): print("Generation complete")
                    results.append(pool.apply_async(f, (y, yIndex, xStart, xEnd, xStepSize, poly,)))

                pool.close()
                pool.join()
                for r in results:
                    res = r.get()
                    for p in res:
                        xIndex = p[0]
                        yIndex = p[1]
                        val = p[2]
                        points[yIndex][xIndex] = val

        else:
            start = time.perf_counter()
            for x in np.arange(xStart, xEnd, xStepSize):
                xIndex = int((x-xStart)/xStepSize)
                print(round((xIndex/xRes)*100,2), "% Completed")
                for y in np.arange(yStart, yEnd, yStepSize):
                    yIndex = int(yRes-1-(y-yStart)/yStepSize)
                    root = findRoot(complex(x,y), poly)
                    if root:
                        for r in range(len(poly.roots)):
                            if math.isclose(root.real, poly.roots[r].real, abs_tol=1e-1) and math.isclose(root.imag, poly.roots[r].imag, abs_tol=1e-1):
                                points[yIndex][xIndex] = r

        print("100% Completed")
        timeTaken = time.perf_counter() - start
        hrs = math.trunc(timeTaken/(60**2))
        mins = math.trunc((timeTaken - hrs*60**2)/60)
        secs = math.trunc((timeTaken - hrs*60**2 - mins*60))
        print("Time taken: ", hrs,"hours,",mins,"minutes and",secs,"seconds")
        if input("Save? y/n;    ") == "y":
            fn = input("Fractal name;   ")
            np.savetxt(fn + ".txt", points)


    top = cm.get_cmap('Oranges_r', 128)
    bottom = cm.get_cmap('Blues', 128)
    newcolors = np.vstack((top(np.linspace(0, 1, 128)),
                        bottom(np.linspace(0, 1, 128))))
    newcmp = ListedColormap(newcolors, name='OrangeBlue')

    ppi = math.sqrt((xRes**2 + yRes**2)/23)

    done = False
    while not done:
        cmap = input("Enter a colour map (v to view all cmaps);   ")
        if cmap == "v":
            print(plt.colormaps())
        else:
            if cmap == "OrangeBlue":
                cmap = newcmp
            fig = plt.matshow(points, cmap=cmap, fignum=1, aspect='auto')
            plt.axis('off')
            plt.minorticks_off()
            plt.show()
            cont = input("Enter a different colour map? y/n;   ")
            if cont != "y":
                done = True
    if input("Save fractal as image? y/n;   ") == "y":
        plt.axis('off')
        plt.minorticks_off()
        plt.imsave(input("Enter a name for the fractal;   ")+".png", points, dpi=ppi, cmap=cmap)
