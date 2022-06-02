class Polynomial:
    roots = []
    coefficients = []
    order = 0
    derivative = None

    def __init__(self, roots=False, coefficients=False):
        if roots:
            self.order = len(roots)
            self.roots = roots
        elif coefficients:
            self.order = len(coefficients) - 1
            self.coefficients = coefficients

    def genCoefficients(self):
        coefficients = [1.0]
        for i in range(self.order):
            coefficients.append(
                ((-1) ** (i + 1) * self.calcSumOfProducts(self.roots, i)).real
            )
        self.coefficients = coefficients

    def genDerivative(self):
        derivCoes = []
        for i in range(len(self.coefficients) - 1):
            derivCoes.append((len(self.coefficients) - 1 - i) * self.coefficients[i])
        self.derivative = Polynomial(coefficients=derivCoes)

    def calc(self, x):
        result = self.coefficients[0]
        for c in range(1, self.order + 1):
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
                sum = sum + x * a

            return sum
        else:
            sum = 0
            for i in range(len(arr)):
                sum = sum + self.calcSumOfProducts(arr[i + 1 :], fix - 1, x * arr[i])
            return sum
