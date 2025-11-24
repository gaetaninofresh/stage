from graph_tool import Graph
import tqdm
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

    # print([f['name'] for f in fs])
    g = gm.make_call_graph(d.r, fs, True)
    gm.save_graph(g, name='test_closed', mode='call', path=f'{out}/graphs')

    g = gm.make_call_graph(d.r, d.enum_f(), False)
    gm.save_graph(g, name='test', mode='call', path=f'{out}/graphs')

    for f in fs:
        if not re.match(r'sym\.imp\.*', f['name']):
            d.decompile_func(f['addr'], save_location=f'{
                             out}/decompiled_funcs')

            bbs = d.func_basic_blocks(d.disasm_function(f['addr']))
            print('-'*80)

            print(f['name'])
            for b in bbs:
                pprint(b.start)
                pprint(b)
                pprint(b.jump_inst())
                pprint(b.end)
                g = gm.make_logic_graph(bbs)
            gm.save_graph(g, name=f'{f['name']}_logic_graph',
                          mode='logic', path=f'{out}/graphs')
