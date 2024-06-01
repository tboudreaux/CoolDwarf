class KramerOpac:
    def __init__(self, X, Z):
        self.X = X
        self.Z = Z

    def kappa(self, temp, density):
        c0 = 4e25
        c1 = (self.Z*(1+self.X))/2
        c2 = density * (temp**(-3.5))
        return c0*c1*c2
