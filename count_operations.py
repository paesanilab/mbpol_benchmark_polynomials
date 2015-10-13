from __future__ import print_function
from glob import glob

for filename in glob("kernels/*polynomial.cu"):
    with open(filename) as f:
        source = f.read()
    print()
    print(filename)
    for symbol in ["+", "*", "=", "pow"]:
        print(symbol, source.count(symbol))
