from typing import List, Dict, Any, Optional, Literal
from r2pipe import open_sync
from decompile import Decompiler, BasicBlock, Instruction
from graph_tool.all import *
import json


class Grapher:

    _default_logic_style = {
        "shape": "box",
        "style": "filled",
        "fillcolor": "white",
        "color": "black",
        "fontname": "Helvetica",
        "fontsize": "12",
        "width": "1",
        "height": "1",
        "margin": "0.12, 0.12",
        "labelloc": "c",
        "labeljust": "l",
        "align": "l"
    }

    def make_call_graph(
        self,
        r: open_sync.open,
        target_fs: list,
        only_target_ref=True
    ):

        g = Graph(directed=True)

        f_name = g.new_vp('string')
        f_addr = g.new_vp('int')
        v_label = g.new_vp('string')

        call_addr = g.new_ep('int')
        e_label = g.new_ep('string')

        g.vertex_properties['f_name'] = f_name
        g.vertex_properties['f_addr'] = f_addr
        g.vertex_properties['label'] = v_label

        g.edge_properties['call_addr'] = call_addr
        g.edge_properties['label'] = e_label

        v_index = {f_addr[v]: v for v in g.vertices()}
        e_index = {call_addr[e]: e for e in g.edges()}

        valid_addrs = {i['addr'] for i in target_fs}

        for f in target_fs:
            func_addr = f['addr']

            if func_addr not in v_index.keys():
                f_v = g.add_vertex()

                f_name[f_v] = f['name']
                f_addr[f_v] = func_addr

                v_index[func_addr] = f_v

            else:
                f_v = v_index[func_addr]

            v_label[f_v] = f'{hex(f_addr[f_v])} : {f_name[f_v]}'

            r.cmd(f's @ {func_addr}')
            xrefs = json.loads(r.cmd('afxj') or '[]')

            i = 0
            for xref in [x for x in xrefs if x['type'] == 'CALL']:
                callee_addr = xref['to']
                _call_addr = xref['from']
                if only_target_ref and callee_addr not in valid_addrs:
                    continue
                else:
                    if callee_addr not in v_index.keys():
                        c_v = g.add_vertex()
                        f_addr[c_v] = callee_addr

                        r.cmd(f's @ {callee_addr}')
                        f_info = json.loads(r.cmd(f'afij') or '[]')

                        if f_info == []:
                            call_name = f'no_name_f_{++i}'
                        else:
                            call_name = f_info[0].get('name')
                        f_name[c_v] = call_name

                        v_index[callee_addr] = c_v
                    else:
                        c_v = v_index[callee_addr]

                    e = g.add_edge(f_v, c_v)
                    e_index[_call_addr] = e
                    call_addr[e] = _call_addr
                    e_label[e] = f'{hex(_call_addr)}'
        return g

    def make_logic_graph(self,
                         bbs: List[BasicBlock]
                         ) -> Graph:
        g = Graph(directed=True)

        start_addr = g.new_vp('int')
        end_addr = g.new_vp('int')
        disasm = g.new_vp('string')
        v_label = g.new_vp('string')

        cond = g.new_ep('string')
        state = g.new_ep('bool')
        e_label = g.new_ep('string')

        g.vertex_properties['start_addr'] = start_addr
        g.vertex_properties['end_addr'] = end_addr
        g.vertex_properties['disasm'] = disasm
        g.vertex_properties['label'] = v_label

        g.edge_properties['cond'] = cond
        g.edge_properties['state'] = state
        g.edge_properties['label'] = e_label

        for b in bbs:
            b_v = g.add_vertex()
            start_addr[b_v] = b.start
            end_addr[b_v] = b.end
            disasm[b_v] = '\n'.join(
                [f'{hex(i.addr)} | {i.disasm}' for i in b.instructions])
            v_label[b_v] = f'{hex(start_addr[b_v])}\n{disasm[b_v]}'

        v_map = {start_addr[v]: v for v in g.vertices()}

        for b in bbs:
            bv = v_map[b.start]
            j_i = b.jump_inst()

            if j_i is not None:
                if j_i.jump in v_map.keys():
                    bv_j = v_map[j_i.jump]
                else:
                    bv_j = g.add_vertex()
                    start_addr[bv_j] = j_i.jump
                    v_label[bv_j] = hex(start_addr[bv_j])
                e = g.add_edge(bv, bv_j)

                if j_i.jump_fail is not None:
                    state[e] = True
                    e_label[e] = f'{state[e]}'
                    if j_i.jump_fail in v_map.keys():
                        bv_jf = v_map[j_i.jump_fail]
                    else:
                        bv_jf = g.add_vertex()
                        start_addr[bv_jf] = j_i.jump_fail
                        v_label[bv_jf] = hex(start_addr[bv_jf])
                    e = g.add_edge(bv, bv_jf)
                    state[e] = False
                    e_label[e] = f'{state[e]}'
        return g

    def save_graph(self, g: Graph, name, mode: Literal['call', 'logic'], path='./graphs'):
        '''
        Saves given graph to path as a .png and as a .dot
        '''

        g.save(f'{path}/{name}.dot', fmt='dot')

        vp = g.vertex_properties
        ep = g.edge_properties
        if mode == 'call':
            graphviz_draw(g, size=(100, 100), eprops=ep, vprops=vp,
                          output=f"{path}/{name}.png", layout="dot")
        elif mode == 'logic':
            gv = g.copy()
            for opt in self._default_logic_style.keys():
                p = gv.new_vp('string')
                gv.vertex_properties[opt] = p
                for v in g.vertices():
                    p[v] = self._default_logic_style[opt]
            graphviz_draw(gv, eprops=ep, vprops=gv.vertex_properties, output=f"{
                          path}/{name}.png", layout="dot")
