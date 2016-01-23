import math
import urllib


class Solver:
    def demo(self):
        a = int(input("a "))
        b = int(input("b "))
        c = int(input("c "))
        d = a + b + c
        disc = math.sqrt(d)
        print(disc)

Solver().demo()
