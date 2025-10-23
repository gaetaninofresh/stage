import r2pipe as r2p
import sys
import json
import argparse
import os
from graph_tool.all import *
from typing import Union
from pathlib import Path
from tqdm import tqdm


# TODO
# - Class analysis
# - Sections analysis
# - Disassemble/Decompile


def r2_info(bin: Path, out_dir: Path):
    r = r2p.open(str(bin.resolve()), flags=["-2"])  # start in silent mode
    r.cmd("-AA")

    r.cmd(f"ij > {out_dir.resolve()}/info.json")  # similar to readelf

    r.cmd(f"izj > {out_dir.resolve()}/strings.json")  # get str in data sect
    r.cmd(f"iHj > {out_dir.resolve()}/headers.json")  # header info
    r.cmd(f"iE > {out_dir.resolve()}/exports.json")  # export info


def r2_vtables(bin: Path, out_dir: Path):
    r = r2p.open(str(bin.resolve()), flags=["-2"])  # start in silent mode
    r.cmd("-AA")

    r.cmd('avj > vtables.json')


def r2_flags(bin: Path, out_dir: Path):
    r = r2p.open(str(bin.resolve()), flags=["-2"])  # start in silent mode
    r.cmd("-AA")

    flags = json.loads(str(r.cmd("fsj")))  # get all flagspaces

    """
    r2 default flagspaces:
        - classes
        - format
        - functions
        - imports
        - registers
        - relocs
        - sections
        - segments
        - strings
        - symbols
    """
    for i in range(0, len(flags)):
        flag = flags[i]["name"]
        r.cmd(f"fs {flag}; fj > {out_dir.resolve()}/{flag}.json")


def r2_disassemble(bin: Path | Path, out_dir: Path):

    r = r2p.open(str(bin.resolve()), flags=["-2"])  # start in silent mode
    r.cmd("-AA")

    # Disassemble everything in the function flagspace
    r.cmd(f'pd@@@F > {out_dir.resolve()}/disasm_readable.asm')


def r2_decompile(bin: Path | Path, out_dir: Path):

    r = r2p.open(str(bin.resolve()), flags=["-2"])  # start in silent mode
    r.cmd("-AA")

    # Decompile everything in the function flagspace
    r.cmd(f'pdg@@@F > {out_dir.resolve()}/decomp.c')


def r2_func_anal(bin, out_dir):
    r = r2p.open(str(bin.resolve()), flags=["-2"])  # start in silent mode
    r.cmd("-AA")
    r.cmd(f"afllj > {out_dir.resolve()}/funcs.json")
    with open(f'{out_dir.resolve()}/funcs.json') as func_json:
        funcs = json.load(func_json)


def r2_func_call_graph(bin, out_dir):
    finfo = None
    r = r2p.open(str(bin.resolve()), flags=['-2'])
    r.cmd('-AA')

    finfo = json.loads(r.cmd('afllj') or '[]')

    if finfo != '[]':

        g = Graph(directed=True)

        f_name = g.new_vp('string')
        f_addr = g.new_vp('string')
        g.vertex_properties['f_name'] = f_name
        g.vertex_properties['f_addr'] = f_addr

        v_index = {f_addr[v]: v for v in g.vertices()}

        for func in tqdm(finfo, desc='Reconstructing func call tree'):
            name = func.get('name')
            addr = hex(func.get('addr'))
            tqdm.write(f' {addr} : {name}')

            if addr not in v_index.keys():
                f_v = g.add_vertex()
                f_name[f_v] = name
                f_addr[f_v] = addr
                v_index[addr] = f_v
            else:
                f_v = v_index[addr]
            r.cmd(f's @ {addr}')
            blocks = json.loads(r.cmd('agfj') or '[]')

            for block in blocks[0]["blocks"]:
                for op in block['ops']:
                    if op.get('type') == 'call':
                        jump_addr = hex(op.get('jump'))

                        if jump_addr in v_index.keys():
                            c_v = v_index[jump_addr]
                        else:
                            c_v = g.add_vertex()
                            v_index[jump_addr] = c_v

                        f_addr[c_v] = jump_addr
                        f_name[c_v] = json.loads(r.cmd(f'afij {jump_addr}') or '[]')[
                            0].get('name')

                        g.add_edge(f_v, c_v)
                        tqdm.write(f'\t -> {jump_addr} : {f_name[c_v]}')

        for v in g.vertices():
            print(f'{v} : {f_addr[v]} - {f_name[v]}')

        g.save(f'{out_dir}/{bin.name}_call_graph.dot', fmt='dot')
        graphviz_draw(g, size=(100, 100), vprops={'label': f_name, 'f_addr': f_addr, 'f_name': f_name}, output=f"{
                      out_dir}/{bin.name}_graph.png", layout="dot")


# TODO
# - concatenate conditions for sym anal
# - combine with func calls


def r2_logic_graph(bin, out_dir):
    r = r2p.open(str(bin.resolve()), flags=['-2'])
    r.cmd('-AA')

    g = Graph(directed=True)

    c_type = g.new_vp('string')  # cjump
    c_addr = g.new_vp('string')
    c_cond = g.new_vp('string')

    g.vertex_properties['c_type'] = c_type
    g.vertex_properties['c_addr'] = c_addr
    g.vertex_properties['c_cond'] = c_cond

    e_state = g.new_ep('bool')
    g.edge_properties['e_state'] = e_state

    finfo = json.loads(r.cmd('aflj') or '[]')

    v_index = {f_addr[v]: v for v in g.vertices()}

    for f in tqdm(finfo):
        r.cmd(f's @ {f.get('addr')}')
        f_ops = json.loads(r.cmd('agfj') or '[]')[0]

        blocks = f_ops.get('blocks')
        for b in blocks:
            tqdm.write(f'{b.get('ops')[-1].get('disasm')
                          } : {b.get('ops')[-1].get('type')}')
            b_ops = ''
            for op in b.get('ops')[:-1]:
                b_ops += f'{op.get('disam')} ;'

            cond = b.get('ops')[-1].get('disasm')
            addr = b.get('ops')[-1].get('addr')
            j_addr = b.get('ops')[-1].get('jump')

            if addr in v_index.keys():
                v1 = v_index[addr]
            else:
                v1 = g.add_vertex()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('bin', type=str)
    parser.add_argument('-o', "--out", default='./out/',
                        action='store', type=str)
    args = parser.parse_args()
    bin = Path(args.bin)
    dir = Path(args.out)

    if not Path.exists(dir):
        os.mkdir(dir)
'''
    r2_info(bin, dir)
    r2_flags(bin, dir)
    r2_disassemble(bin, dir)
    r2_decompile(bin, dir)
    r2_func_call_graph(bin, dir)
    r2_func_call_graph(bin, dir)
'''
r2_logic_graph(bin, dir)
