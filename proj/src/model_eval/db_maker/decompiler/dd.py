from graph_tool import Graph
from tqdm import tqdm
import os
import argparse
import json
from pathlib import Path
from decompile import Decompiler
from graphmaker import Grapher
import re

from typing import List, Dict, Any, Optional
from pprint import pprint

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('bin', type=str)
    parser.add_argument('-o', "--out", default='./out',
                        action='store', type=str)

    args = parser.parse_args()
    bin = args.bin
    out = args.out

    d = Decompiler(bin)
    gm = Grapher()

    fs = d.filter_funcs(exclude_on_name_re=False)

    g = gm.make_call_graph(d.r, fs, True)
    gm.save_graph(g, name='test_closed', mode='call', path=f'{out}/graphs')

    g = gm.make_call_graph(d.r, d.enum_f(), False)
    gm.save_graph(g, name='test', mode='call', path=f'{out}/graphs')

    for f in tqdm([f for f in fs if not re.match(r'sym\.imp\.*', f['name'])]):
        tqdm.write(f'Building {f['name']} logic graph')
        d.decompile_func(f['addr'], save_location=f'{
            out}/decompiled_funcs')

        bbs = d.func_basic_blocks(d.disasm_function(f['addr']))
        for b in bbs:
            g = gm.make_logic_graph(bbs)
        gm.save_graph(g, name=f'{f['name']}_logic_graph',
                      mode='logic', path=f'{out}/graphs')
