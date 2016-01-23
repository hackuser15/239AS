import math
import urllib


class Solver:
    def demo(self):
        a = int(input("a "))
        b = int(input("b "))
        c = int(input("c "))
        d1 = a + b + c
        disc = math.sqrt(d1)
        print(disc)

Solver().demo()
