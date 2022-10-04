# got to chunk 37 (redo)

import os
import random
import psutil
import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap
import multiprocess as mp
import time
from PIL import Image

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
    maxIter = 1000
    result = poly.calc(x0)
    root = x0
    gradZero = False
    while (
        not (
            math.isclose(result.real, 0, abs_tol=1e-5)
            and math.isclose(result.imag, 0, abs_tol=1e-5)
        )
        and iterCount < maxIter
        and not gradZero
    ):
        root, gradZero = newtonRaphson(root, poly)
        result = poly.calc(root)
        iterCount = iterCount + 1
    if iterCount == maxIter or gradZero:
        if gradZero:
            print("gradient zero")
        if iterCount == maxIter:
            print("maxed iters reached")
        return False, False  # Root not found
    else:
        return root, True  # Root found


def displayPoints(points):
    yesNoInput = Input("y/n")
    xRes = points.shape[1]
    yRes = points.shape[0]
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


def compileChunks(fn, cmap):
    md = open("Saves/" + fn + ".txt", "r")
    numChunks = int(md.readline().strip())
    xRes, yRes = [int(x) for x in md.readline().strip().split(" ")]

    for c in range(numChunks):
        chunk = np.loadtxt("Saves/" + fn + str(c) + ".txt")
        plt.matshow(chunk, cmap=cmap, fignum=1, aspect="auto")
        plt.axis("off")
        plt.minorticks_off()
        plt.imsave(
            "TempSaves/Chunk" + str(c) + ".png",
            chunk,
            dpi=math.sqrt((xRes**2 + yRes**2) / 23),
            cmap=cmap,
        )
    chunks = [Image.open("TempSaves/Chunk" + str(x) + ".png") for x in range(numChunks)]
    chunks = chunks[::-1]
    cMerge = Image.new(chunks[0].mode, (xRes, yRes))
    y = 0
    for c in chunks:
        cMerge.paste(c, (0, y))
        y += c.height
    cMerge.save("Images/" + fn + ".png")
    [os.remove("TempSaves/Chunk" + str(x) + ".png") for x in range(numChunks)]


def genFractalImage(
    poly, xStart, xEnd, yStart, yEnd, xRes, yRes, fName, useMP
):  # saves the fractal as txt and returns the points
    xStepSize = abs((xEnd - xStart) / xRes)
    yStepSize = abs((yEnd - yStart) / yRes)
    points = np.ndarray(shape=(yRes, xRes))

    if useMP:

        def f(y, yIndex, xStart, xEnd, xStepSize, poly):
            res = []
            for x in np.arange(xStart, xEnd, xStepSize):
                xIndex = int((x - xStart) / xStepSize)
                root, rootFound = findRoot(complex(x, y), poly)
                if rootFound:
                    rootInList = False
                    for r in range(len(poly.roots)):
                        if math.isclose(
                            root.real, poly.roots[r].real, abs_tol=1e-2
                        ) and math.isclose(root.imag, poly.roots[r].imag, abs_tol=1e-2):
                            res.append([xIndex, yIndex, r + 1])
                            rootInList = True
                    if not rootInList:
                        print("Root found but not in root list")

                else:
                    print("Root not found for x: " + str(x) + ", y: " + str(y))
            return res

        results = []

        start = time.perf_counter()
        with mp.Pool(mp.cpu_count()) as pool:
            for y in np.arange(yStart, yEnd, yStepSize):
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
                    points[yIndex][xIndex] = val

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
                            root.real, poly.roots[r].real, abs_tol=1e-2
                        ) and math.isclose(root.imag, poly.roots[r].imag, abs_tol=1e-2):
                            points[yIndex][xIndex] = r

    print("100% Completed")
    timeTaken = time.perf_counter() - start
    hrs = math.trunc(timeTaken / (60**2))
    mins = math.trunc((timeTaken - hrs * 60**2) / 60)
    secs = math.trunc((timeTaken - hrs * 60**2 - mins * 60))
    print("Time taken: ", hrs, "hours,", mins, "minutes and", secs, "seconds")
    np.savetxt("Saves/" + fName + ".txt", points)
    return points


if __name__ == "__main__":
    yesNoInput = Input("y/n")
    floatInput = Input("float")
    intInput = Input("int")

    for d in ["Images", "Saves", "TempSaves"]:
        if not os.path.isdir(d):
            os.mkdir(d)

    if yesNoInput.getInput("Load fractal? y/n"):
        saveName = input("Enter fractal name;   ")
        try:
            with open("Saves/" + saveName + ".txt") as f:
                numChunks = int(f.readline())
                compileChunks(saveName, input("Enter a colour map;    "))
        except:
            points = np.loadtxt("Saves/" + saveName + ".txt")
            displayPoints(points)

    else:
        roots = []
        if yesNoInput.getInput("Enter custom roots? y/n"):
            done = False
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
            noRoots = intInput.getInput(
                "Enter the number of complex root pairs to generate"
            )
            for r in range(noRoots):
                randRe = random.randint(-10, 10) * (-1) ** random.randint(0, 1)
                randIm = random.randint(-10, 10) * (-1) ** random.randint(0, 1)
                roots.append(complex(randRe, randIm))
                if randIm != 0:
                    roots.append(complex(randRe, -1 * randIm))

        xStart = floatInput.getInput("Enter start x bound")
        xEnd = floatInput.getInput("Enter end x bound")
        yStart = floatInput.getInput("Enter start y bound")
        yEnd = floatInput.getInput("Enter end y bound")

        print(
            "Standard resolutions: 640x480 (480p), 1280x720 (720p), 1920x1080 (1080p), 2560x1440 (1440p), 3840x2160 (4K), 7680x4320 (8K), 15360×8640 (16K), 30720x17280 (32K), 61440x34560 (64K), 122880x69120 (128K)"
        )
        xRes = intInput.getInput("Enter x resolution")
        yRes = intInput.getInput("Enter y resolution")

        poly = Polynomial(roots=roots)
        poly.genCoefficients()
        print("Roots: ", poly.roots)
        print("Coefficients: ", poly.coefficients)
        poly.genDerivative()

        fName = input("Enter a name for the fractal;    ")
        useMP = yesNoInput.getInput("Use multiprocessing? y/n")

        memory = psutil.virtual_memory().available
        arrSize = (yRes * xRes * 64) / 8
        start = None
        if arrSize > memory * 0.5:
            print(
                "Insuffient memory to generate image in one go, must save now and compile later"
            )
            xStepSize = abs((xEnd - xStart) / xRes)
            yStepSize = abs((yEnd - yStart) / yRes)
            with open("Saves/" + fName + ".txt", "wt") as save:  # main save file
                start = time.perf_counter()
                rowsCount = 540 * (36 - 1)
                cYRes = math.trunc(
                    (7680 * 4320) / (xRes)
                )  # float64 used for each array index
                cCount = rowsCount / cYRes
                numChunks = math.ceil(yRes / cYRes)
                save.write(str(numChunks) + "\n")
                save.write(str(xRes) + " " + str(yRes) + "\n")
                save.flush()
                while rowsCount < yRes:
                    cYStart = (
                        int((yStart + rowsCount * yStepSize) / yStepSize) * yStepSize
                    )
                    cYEnd = (
                        int(
                            (yStart + rowsCount * yStepSize + cYRes * yStepSize)
                            / yStepSize
                        )
                        * yStepSize
                    )
                    if cYEnd > yEnd:
                        cYEnd = yEnd

                    rowsCount = rowsCount + cYRes

                    rows = cYRes
                    if rowsCount > yRes:
                        rows = cYRes - (rowsCount - yRes)

                    save.write(str(rows) + " " + str(xRes) + "\n")
                    save.flush()
                    print(
                        "Processing chunk "
                        + str(math.ceil(rowsCount / cYRes))
                        + "/"
                        + str(numChunks)
                    )

                    genFractalImage(
                        poly,
                        xStart,
                        xEnd,
                        cYStart,
                        cYEnd,
                        xRes,
                        cYRes,
                        fName + str(cCount),
                        useMP,
                    )
                    cCount += 1
                save.close()

            print("All Chunks 100% Completed")
            timeTaken = time.perf_counter() - start
            hrs = math.trunc(timeTaken / (60**2))
            mins = math.trunc((timeTaken - hrs * 60**2) / 60)
            secs = math.trunc((timeTaken - hrs * 60**2 - mins * 60))
            print(
                "Total Time taken: ",
                hrs,
                "hours,",
                mins,
                "minutes and",
                secs,
                "seconds",
            )
            print(
                "As fractal is too big to be loaded into RAM, entering a cmap will save it as an image"
            )

            top = cm.get_cmap("Oranges_r", 128)
            bottom = cm.get_cmap("Blues", 128)
            newcolors = np.vstack(
                (top(np.linspace(0, 1, 128)), bottom(np.linspace(0, 1, 128)))
            )
            newcmp = ListedColormap(newcolors, name="OrangeBlue")

            done = False
            while not done:
                cmap = input("Enter a colour map (v to view all cmaps);   ")
                if cmap == "v":
                    print(plt.colormaps())
                else:
                    if cmap == "OrangeBlue":
                        cmap = newcmp
                    compileChunks(fName, cmap=cmap)
                    if not yesNoInput.getInput("Choose a different colour map? y/n"):
                        done = True

        else:
            displayPoints(
                genFractalImage(
                    poly, xStart, xEnd, yStart, yEnd, xRes, yRes, fName, useMP
                )
            )
