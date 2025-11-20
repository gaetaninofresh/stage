import r2pipe as r2
import json
import re
from pathlib import Path
import os


# CURRENT FEATURES

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
        # if os.access(bin, os.X_OK):
        #      raise PermissionError(f'Path is not executable: {bin}')

        self.bin = bin

        self.r = r2.open(bin, flags=["-2", "-e bin.relocs.apply=true"])
        self.r.cmd('-AA')

    def filter_funcs_by_name(self, funcs, pattern=None):
        '''
        Return functions matching an hardcoded regex pattern from provided json function list
        '''

        if pattern == None:
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

    # Lazy implementation but seems to work
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

    def filter_funcs(self,
                     exclude_plt=True,
                     exclude_no_xref=True,
                     exclude_on_name_re=True,
                     f_name_pattern=None,
                     r2_func_rename=False
                     ):
        '''
        Applies selected filters to the full function list and returns the 'good' ones
        '''

        f = self.enum_f()
        rem_f = []
        if exclude_plt:
            rem_f.extend(self.plt_sym_f(f))
        if exclude_no_xref:
            rem_f.extend(self.no_xref_f(f))
        if exclude_on_name_re:
            rem_f.extend(self.filter_funcs_by_name(f, pattern=f_name_pattern))
        if r2_func_rename:
            self.clean_f_name(f)

        rem_f_keys = {i['addr'] for i in rem_f}
        return [i for i in f if i['addr'] not in rem_f_keys]

    def decompile_func(self,
                       f_addr: int,
                       file_name=None,
                       save_location=None,
                       ):
        self.r.cmd(f's @ {f_addr}')

        if save_location is not None:
            if file_name is None:
                finfo = json.loads(self.r.cmd('afij') or '[]')
                file_name = finfo[0]['name'] + '.c'
                print(f'{save_location}/{file_name}')
            self.r.cmd(f'pdg > {save_location}/{file_name}')
        else:
            return self.r.cmd('pdg')
