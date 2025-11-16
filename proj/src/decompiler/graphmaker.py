from r2pipe import open_sync
from decompile import Decompiler
from graph_tool.all import *
import json
import r2pipe as r2


class Grapher:
    def make_graph(
        self,
        r: open_sync.open,
        target_fs: list,
        save: bool
    ):

        g = Graph(directed=True)

        f_name = g.new_vp('string')
        f_addr = g.new_vp('string')
        g.vertex_properties['f_name'] = f_name
        g.vertex_properties['f_addr'] = f_addr

        v_index = {f_addr[v]: v for v in g.vertices()}

        for f in target_fs:
            func_addr = f['addr']

            if func_addr not in v_index.keys():
                f_v = g.add_vertex()

                f_name[f_v] = f['name']
                f_addr[f_v] = func_addr

                v_index[func_addr] = f_v

            else:
                f_v = v_index[func_addr]

            r.cmd(f's @ {func_addr}')
            xrefs = json.loads(r.cmd('afxj') or '[]')

            i = 0
            for xref in [x for x in xrefs if x['type'] == 'CALL']:
                call_addr = xref['to']

                if call_addr not in v_index.keys():
                    c_v = g.add_vertex()

                    f_addr[c_v] = call_addr
                    r.cmd(f's @ {call_addr}')
                    f_info = json.loads(r.cmd(f'afij') or '[]')

                    print(f_info)

                    if f_info == []:
                        call_name = f'no_name_{++i}'
                    else:
                        call_name = f_info[0].get('name')
                    f_name[c_v] = call_name

                    v_index[call_addr] = c_v
                else:
                    c_v = v_index[call_addr]

                g.add_edge(f_v, c_v)

        # for v in g.vertices():
        #    print(f'{v} : {f_addr[v]} - {f_name[v]}')
        out_dir = './graphs/'

        g.save(f'{out_dir}/call_graph.dot', fmt='dot')
        graphviz_draw(g, size=(300, 300), vprops={'label': f_name, 'f_addr': f_addr, 'f_name': f_name}, output=f"{
                      out_dir}/graph.png", layout="dot")


if __name__ == '__main__':
    g = Grapher()
    d = Decompiler('./bin')
    g.make_graph(d.r, d.enum_f(), True)
