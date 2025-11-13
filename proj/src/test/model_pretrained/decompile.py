import r2pipe as r2
import json
import re

f_name_regex = [

    r"vector__",
    r"\.end__",
    r"__normal_iterator_",
    r"\.insert_",
    r"std::",
    r"_std",
    r"std::allocator",
    r"imp._*",
    r"sym\.imp\.memset",
    r"sym\.imp\.atoi",
    r"sym\.imp\.htons",
    r"sym\.imp\.inet_addr",
    r"method\.std::vector.*?\._vector__",
    r"method\.std::vector.*?vector_.*?",
]


f_name_patt = [re.compile(rx) for rx in f_name_regex]


def filter_funcs_by_name(funcs, strictness=1):
    for f in funcs:
        if any(re.search(p, f['name']) for p in f_name_patt):
            f['rm'] = 1
        else:
            f['rm'] = 0
    f_clean = json.loads(json.dumps([f for f in funcs if f['rm'] == 0]))
    for f in f_clean:
        del f['rm']
    return f_clean


def clean_f_name(r, funcs):
    for f in funcs:
        if '.' in f['name']:
            r.cmd(f's @ {f['addr']}')
            name = re.sub(r'\.', '_', f['name'])
            r.cmd(f'afn {name}')


def no_xref_f(r, funcs):
    rem_f = []
    for f in funcs:
        r.cmd(f' s @ {f['addr']}')
        xrefs = r.cmd('afxj')
        if xrefs == '[]':
            rem_f.append(f)
    

def decompile_filt_f(bin):
    r = r2.open(bin, flags=['-2'])
    r.cmd('-AA')

    funcs = json.loads(r.cmd('aflj') or '[]')
    funcs_filt = filter_funcs_by_name(funcs)
    clean_f_name(r, funcs)

    decompiled = ''

    for f in funcs_filt:
        r.cmd(f's @ {f['addr']}')
        f_dec = r.cmd('pdg') or ''
        decompiled += f_dec

    return decompiled
