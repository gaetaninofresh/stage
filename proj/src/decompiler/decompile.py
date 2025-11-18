import r2pipe as r2
import json
import re
from pathlib import Path
import os


# CURRENT FEATURES

class Decompiler:
    def __init__(self, bin: str):

        if not os.path.exists(bin):
            raise FileNotFoundError(f"Path does not exists: {bin}")
        # if os.access(bin, os.X_OK):
        #      raise PermissionError(f'Path is not executable: {bin}')

        self.bin = bin

        self.f_name_regex = [
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

        self.f_name_patt = [re.compile(rx) for rx in self.f_name_regex]
        self.r = r2.open(bin, flags=["-2", "-e bin.relocs.apply=true"])
        self.r.cmd('-AA')

    def filter_funcs_by_name(self, funcs, strictness=1):
        '''
        Return functions matching an hardcoded regex pattern from provided json function list
        '''
        rem_f = []
        for f in funcs:
            if any(re.search(p, f["name"]) for p in self.f_name_patt):
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
