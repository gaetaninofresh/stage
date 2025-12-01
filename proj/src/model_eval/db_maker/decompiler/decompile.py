import r2pipe as r2
import json
import re
from pathlib import Path
import os
from dataclasses import dataclass
from typing import List, Dict, Any, Literal, Optional


@dataclass
class Instruction:
    addr: int
    opcode: str
    disasm: str
    size: int
    type: str
    jump: Optional[int] = None
    jump_fail: Optional[int] = None


@dataclass
class BasicBlock:
    instructions: List[Instruction]

    @property
    def start(self):
        return self.instructions[0].addr

    @property
    def end(self):
        i = self.instructions[-1]
        return i.addr + i.size

    @property
    def size(self):
        return self.end - self.start

    def jump_inst(self):
        i = self.instructions[-1]
        return i if i.jump is not None or i.type == 'call' else None


class Decompiler:
    _default_f_name_regex = [
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

    _default_f_name_patt = [re.compile(rx) for rx in _default_f_name_regex]

    def __init__(self, bin: str):

        if not os.path.exists(bin):
            raise FileNotFoundError(f"Path does not exists: {bin}")

        self.bin = bin

        self.r = r2.open(bin, flags=["-2", "-e bin.relocs.apply=true"])
        self.r.cmd('-AA')

    def close(self):
        try:
            if self.r:
                self.r.quit()
                self.r = None
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def filter_funcs_by_name(self, funcs, pattern=None):
        '''
        Return functions matching an hardcoded regex pattern from provided json function list
        '''

        if pattern is None:
            pattern = self._default_f_name_patt

        rem_f = []
        for f in funcs:
            if any(re.search(p, f["name"]) for p in pattern):
                rem_f.append(f)
        return rem_f

    def clean_f_name(self, funcs):
        '''
        Renames provided functions inside r2 to avoid errors in postprocessing by changin . to _
        '''
        for f in funcs:
            if "." in f["name"]:
                self.r.cmd(f's @ {f['addr']}')
                name = re.sub(r"\.", "_", f["name"])
                self.r.cmd(f"afn {name}")

    def no_xref_f(self, funcs):
        '''
        Return functions with no xrefs from provided json function list
        '''
        rem_f = []
        for f in funcs:
            self.r.cmd(f's @ {f['addr']}')
            xrefs = json.loads(str(self.r.cmd("afxj")))
            if xrefs == []:
                rem_f.append(f)
        return rem_f

    def enum_f(self):
        '''
        Returns a json object containing function infos from r2
        '''
        fs = json.loads(str(self.r.cmd('aflj')))
        return fs

    def plt_sym_f(self, funcs):
        '''
        Return decompiler "artifact" functions consisting of an indirect call to PLT
        from provided json function list (they have a very good reason to exist but they're just noise atm)
        '''

        sec = json.loads(str(self.r.cmd('iSj')))
        plt_sec = [s for s in sec if re.match(r'\.plt\w*', s['name'])]

        plt_start = plt_sec[0]['paddr']
        plt_end = plt_sec[-1]['paddr']
        plt_f = []
        for f in funcs:
            if f['addr'] in range(plt_start, plt_end):
                plt_f.append(f)
        return plt_f

    def sym_imp_wrapper_f(self, funcs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        '''
        Returns all functions that
        '''
        return []

    def filter_funcs(self,
                     func_list: List[Dict] | None = None,
                     exclude_plt: bool = True,
                     exclude_no_xref: bool = True,
                     exclude_on_name_re: bool = True,
                     f_name_pattern=None,
                     ):
        '''
        Applies selected filters to the given function list and returns the 'good' ones
        '''

        if func_list is None:
            func_list = self.enum_f()

        rem_f = []
        if exclude_plt:
            rem_f.extend(self.plt_sym_f(func_list))
        if exclude_no_xref:
            rem_f.extend(self.no_xref_f(func_list))
        if exclude_on_name_re:
            rem_f.extend(self.filter_funcs_by_name(
                func_list, pattern=f_name_pattern))

        rem_f_keys = {i['addr'] for i in rem_f}
        return [i for i in func_list if i['addr'] not in rem_f_keys]

    def decompile_func(self,
                       f_addr: int,
                       file_name: str | None = None,
                       format: Literal['raw', 'json'] = 'raw',
                       save_location: str | None = None,
                       ):
        self.r.cmd(f'ss {f_addr}')

        if save_location is not None:
            if file_name is None:
                finfo = json.loads(self.r.cmd('afij') or '[]')
                file_name = finfo[0]['name'] + '.c'
            self.r.cmd(f'pdg{'j' if format == 'json' else ''} > {
                       save_location}/{file_name}')

        if format == 'json':
            return json.loads(self.r.cmd('pdgj') or '[]')
        else:
            return self.r.cmd('pdg')

    def disasm_function(self, f_addr: int):
        return json.loads(self.r.cmd(f'ss {f_addr}; pdfj') or '[]')

    def func_basic_blocks(self, disasm: Dict[str, Any]):
        ops = disasm['ops']
        ops.sort(key=lambda x: x['addr'])

        op_map = {op['addr']: op for i, op in enumerate(ops)}
        addr_map = {op['addr']: i for i, op in enumerate(ops)}

        stack = [0]
        seen = []
        bbs = []

        while len(stack) > 0:
            b = BasicBlock([])
            i = stack.pop()
            seen.append(i)
            for op in ops[i:]:
                ins = Instruction(
                    addr=op['addr'],
                    disasm=op['disasm'],
                    opcode=op['opcode'],
                    size=op['size'],
                    type=op['type'],
                    jump=op['jump'] if 'jump' in op.keys() else None,
                    jump_fail=op['fail'] if 'fail' in op.keys() else None
                )

                b.instructions.append(ins)

                if ins.jump is not None and ins.type != 'call':
                    if addr_map[ins.jump] not in stack and addr_map[ins.jump] not in seen:
                        stack.append(addr_map[ins.jump])
                        seen.append(addr_map[ins.jump])
                        if ins.jump_fail is not None and addr_map[ins.jump_fail] not in seen and ins.jump_fail not in stack:
                            stack.append(addr_map[ins.jump_fail])
                            seen.append(addr_map[ins.jump_fail])
                    break
                if i+1 < len(ops) and addr_map[ops[i+1]['addr']] in stack:
                    break
            bbs.append(b)
        return bbs

    def relevant_fs(self, check_sec_calls: bool = False):
        xrefs = json.loads(self.r.cmd('ss main; afxj') or '[]')
        primary_f_call = [f for f in xrefs if f['type'] == 'CALL'][-2]
        if check_sec_calls:
            secondary_fs_calls = [fun for fun in json.loads(
                self.r.cmd(f'ss {primary_f_call['to']}; afxj') or '[]') if fun['type'] == 'CALL']
            fs = secondary_fs_calls
        else:
            fs = [primary_f_call]
        funcs = []
        for f in fs:
            finfo = json.loads(self.r.cmd(f'ss {f['to']}; afij') or '[]')
            funcs.append(*finfo)
        return self.filter_funcs(func_list=funcs)


if __name__ == '__main__':
    r = Decompiler(
        './test_db_mini/good/CWE121_Stack_Based_Buffer_Overflow__CWE85_int64_t_alloca_memcpy7-good')
    print(r.relevant_fs())
