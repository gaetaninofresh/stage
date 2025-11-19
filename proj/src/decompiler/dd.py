from graph_tool import Graph
import tqdm
import os
import argparse
import json
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
    gm = Grapher()

    fs = d.filter_funcs(exclude_on_name_re=False)

    print([f['name'] for f in fs])
    g = gm.make_call_graph(d.r, fs, True)
    gm.save_graph(g, 'test_closed')

    g = gm.make_call_graph(d.r, d.enum_f(), False)
    gm.save_graph(g, 'test')
