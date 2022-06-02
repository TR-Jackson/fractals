import os
import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap
import multiprocess as mp
import time

from MyPolynomial import Polynomial
from MyInput import Input


def newtonRaphson(x, poly):
    fPrime = poly.calcDerivative(x)
    f = poly.calc(x)
    if fPrime == 0:
        return x, True  # Won't find root
    return x - f / fPrime, False


def findRoot(x0, poly):
    iterCount = 0
    maxIter = 100
    result = poly.calc(x0)
    root = x0
    diverges = False
    while (
        not (
            math.isclose(result.real, 0, abs_tol=1e-2)
            and math.isclose(result.imag, 0, abs_tol=1e-2)
        )
        and iterCount < maxIter
        and not diverges
    ):
        root, diverges = newtonRaphson(root, poly)
        result = poly.calc(root)
        iterCount = iterCount + 1
    if iterCount == maxIter or diverges:
        return False  # Root not found
    else:
        return root  # Root found


if __name__ == "__main__":
    yesNoInput = Input("y/n")
    floatInput = Input("float")
    intInput = Input("int")

    for d in ["Images", "Saves", "TempSaves"]:
        if not os.path.isdir(d):
            os.mkdir(d)

    if yesNoInput.getInput("Load fractal? y/n"):
        points = np.loadtxt("Saves/" + input("Enter fractal name;   ") + ".txt")
        xRes = points.shape[1]
        yRes = points.shape[0]
    else:
        if yesNoInput.getInput("Enter custom roots? y/n"):
            done = False
            roots = []
            rootInput = Input("float")
            while not done:
                real = rootInput.getInput("Enter the real part of the root")
                imag = rootInput.getInput("Enter the imaginary part of the root")
                roots.append(complex(real, imag))
                if imag != 0:
                    roots.append(complex(real, -imag))
                if not yesNoInput.getInput("Add another root? y/n"):
                    done = True
        else:
            # roots = [complex(0,1), complex(0,-1), complex(1,0), complex(math.sqrt(2),math.sqrt(3)), complex(math.sqrt(2),-math.sqrt(3)), complex(1,math.sqrt(3)), complex(1,-math.sqrt(3))]
            roots = [
                complex(1, 1),
                complex(1, -1),
                complex(0, 1),
                complex(0, -1),
                complex(0, 0),
            ]

        xStart = floatInput.getInput("Enter start x bound")
        xEnd = floatInput.getInput("Enter end x bound")
        yStart = floatInput.getInput("Enter start y bound")
        yEnd = floatInput.getInput("Enter end y bound")

        print(
            "Standard resolutions: 640x480, 1280x720, 1920x1080, 2560x1440, 3840x2160, 7680x4320"
        )
        xRes = intInput.getInput("Enter x resolution")
        yRes = intInput.getInput("Enter y resolution")

        xStepSize = abs((xEnd - xStart) / xRes)
        yStepSize = abs((yEnd - yStart) / yRes)

        poly = Polynomial(roots=roots)
        poly.genCoefficients()
        print(poly.roots)
        print(poly.coefficients)
        poly.genDerivative()

        points = np.ndarray(shape=(yRes, xRes))

        if yesNoInput.getInput("Use multiprocessing? y/n"):
            chunk = np.ndarray(shape=(yRes, xRes))
            numChunks = 1
            if yesNoInput.getInput(
                "Perform in chunks to avoid running out of RAM for high resolution images? y/n"
            ):
                memorySize = floatInput.getInput(
                    "How many GB of RAM available? (check msinfo32)"
                )
                numChunks = (
                    math.ceil((xRes * yRes * 32) / (memorySize * (10**9) * 8)) + 1
                )

            def f(y, yIndex, xStart, xEnd, xStepSize, poly):
                res = []
                for x in np.arange(xStart, xEnd, xStepSize):
                    xIndex = int((x - xStart) / xStepSize)
                    root = findRoot(complex(x, y), poly)
                    if root:
                        for r in range(len(poly.roots)):
                            if math.isclose(
                                root.real, poly.roots[r].real, abs_tol=1e-1
                            ) and math.isclose(
                                root.imag, poly.roots[r].imag, abs_tol=1e-1
                            ):
                                res.append([xIndex, yIndex, r])
                return res

            results = []

            start = time.perf_counter()
            for c in range(numChunks):
                with mp.Pool(mp.cpu_count()) as pool:
                    # print(
                    #     "Processing chunk "
                    #     + str(c + 1)
                    #     + " from "
                    #     + str(yStart + yRes * (c / numChunks) * yStepSize)
                    #     + " to "
                    #     + str(
                    #         yStart
                    #         + yRes * ((c + 1) / numChunks) * yStepSize
                    #         - yStepSize
                    #     )
                    # )
                    if numChunks > 1:
                        print("Processing chunk " + str(c + 1) + "/" + str(numChunks))

                    for y in np.arange(
                        yStart + yRes * (c / numChunks) * yStepSize,
                        yStart + yRes * ((c + 1) / numChunks) * yStepSize,
                        yStepSize,
                    ):
                        yIndex = int(yRes - 1 - (y - yStart) / yStepSize)
                        results.append(
                            pool.apply_async(
                                f,
                                (
                                    y,
                                    yIndex,
                                    xStart,
                                    xEnd,
                                    xStepSize,
                                    poly,
                                ),
                            )
                        )
                    pool.close()
                    pool.join()
                    for r in results:
                        res = r.get()
                        for p in res:
                            xIndex = p[0]
                            yIndex = p[1]
                            val = p[2]
                            if numChunks > 1:
                                chunk[yIndex][xIndex] = val
                            else:
                                points[yIndex][xIndex] = val

                    if numChunks > 1:
                        np.savetxt("TempSaves/Chunk" + str(c) + ".txt", chunk)

            if numChunks > 1:
                for c in range(numChunks - 1, -1, -1):
                    chunk = np.loadtxt("TempSaves/Chunk" + str(c) + ".txt")
                    points = points + chunk
                    os.remove("TempSaves/Chunk" + str(c) + ".txt")

        else:
            start = time.perf_counter()
            for x in np.arange(xStart, xEnd, xStepSize):
                xIndex = int((x - xStart) / xStepSize)
                print(round((xIndex / xRes) * 100, 2), "% Completed")
                for y in np.arange(yStart, yEnd, yStepSize):
                    yIndex = int(yRes - 1 - (y - yStart) / yStepSize)
                    root = findRoot(complex(x, y), poly)
                    if root:
                        for r in range(len(poly.roots)):
                            if math.isclose(
                                root.real, poly.roots[r].real, abs_tol=1e-1
                            ) and math.isclose(
                                root.imag, poly.roots[r].imag, abs_tol=1e-1
                            ):
                                points[yIndex][xIndex] = r

        print("100% Completed")
        timeTaken = time.perf_counter() - start
        hrs = math.trunc(timeTaken / (60**2))
        mins = math.trunc((timeTaken - hrs * 60**2) / 60)
        secs = math.trunc((timeTaken - hrs * 60**2 - mins * 60))
        print("Time taken: ", hrs, "hours,", mins, "minutes and", secs, "seconds")
        if yesNoInput.getInput("Save? y/n"):
            np.savetxt("Saves/" + input("Fractal name;   ") + ".txt", points)

    top = cm.get_cmap("Oranges_r", 128)
    bottom = cm.get_cmap("Blues", 128)
    newcolors = np.vstack((top(np.linspace(0, 1, 128)), bottom(np.linspace(0, 1, 128))))
    newcmp = ListedColormap(newcolors, name="OrangeBlue")

    ppi = math.sqrt((xRes**2 + yRes**2) / 23)

    done = False
    while not done:
        cmap = input("Enter a colour map (v to view all cmaps);   ")
        if cmap == "v":
            print(plt.colormaps())
        else:
            if cmap == "OrangeBlue":
                cmap = newcmp
            fig = plt.matshow(points, cmap=cmap, fignum=1, aspect="auto")
            plt.axis("off")
            plt.minorticks_off()
            plt.show()
            if not yesNoInput.getInput("Choose a different colour map? y/n"):
                done = True
    if yesNoInput.getInput("Save fractal as image? y/n"):
        plt.axis("off")
        plt.minorticks_off()
        plt.imsave(
            "Images/" + input("Enter a name for the fractal;   ") + ".png",
            points,
            dpi=ppi,
            cmap=cmap,
        )
