from graph_tool import Graph
import tqdm
import os
import argparse
from pathlib import Path
from decompile import Decompiler
from graphmaker import Grapher


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('bin', type=str)
    parser.add_argument('-o', "--out", default='./out/',
                        action='store', type=str)

    args = parser.parse_args()
    bin = args.bin
    out = args.out

    d = Decompiler(bin)
    g = Grapher()

    fs = d.enum_f()
