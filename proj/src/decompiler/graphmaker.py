from typing import List, Dict, Any, Optional
from r2pipe import open_sync
from decompile import Decompiler
from graph_tool.all import *
import json
import r2pipe as r2


# TODO:
# - Fix no name functions
# - Fix reloc calls
# Issues might/should be tackled upstream in function feeding
# - Should graph calls without control over r2 but it's already done like this atm...


class Grapher:
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
                         disasm: Dict[str, Any]
                         ) -> Graph:
        '''
        PLACEHOLDER
        '''

        g = Graph(directed=True)

        start_addr = g.new_vp('int')
        end_addr = g.new_vp('int')
        code = g.new_vp('string')
        v_label = g.new_vp('string')

        cond = g.new_ep('string')
        state = g.new_ep('bool')
        e_label = g.new_ep('string')

        g.vertex_properties['start_addr'] = start_addr
        g.vertex_properties['end_addr'] = end_addr
        g.vertex_properties['code'] = code
        g.vertex_properties['label'] = v_label

        g.edge_properties['cond'] = cond
        g.edge_properties['state'] = state
        g.edge_properties['label'] = e_label

        ops = disasm['ops']
        ops.sort(key=lambda x: x['addr'])  # One can never be too sure

        return g

    def save_graph(self, g: Graph, name, path='./graphs'):
        '''
                Saves given graph to path as a .png and as a .dot
                '''
        g.save(f'{path}/{name}.dot', fmt='dot')

        vp = g.vertex_properties
        ep = g.edge_properties

        graphviz_draw(g, size=(100, 100), eprops=ep, vprops=vp, output=f"{
            path}/{name}.png", layout="dot")


if __name__ == '__main__':
    gm = Grapher()
    d = Decompiler('./bin')
    g = gm.make_call_graph(d.r, d.enum_f(), True)
    gm.save_graph(g, 'test')
