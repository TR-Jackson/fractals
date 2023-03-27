class Input:
    type = None
    value = None

    def __init__(self, type):
        self.type = type

    def getInput(self, msg):
        val = None
        while True:
            val = input(msg + ";    ")
            if self.type == "y/n":
                if val.lower() == "y":
                    return True
                if val.lower() == "n":
                    return False
            else:
                try:
                    if self.type == "int":
                        val = int(val)
                    elif self.type == "float":
                        val = float(val)
                    return val
                except:
                    continue
