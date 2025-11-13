import decompile


def full_clean(d: decompile.Decompiler):
    f = d.enum_f()
    no_xref = d.no_xref_f(f)


if __name__ == '__main__':
    d = decompile.Decompiler('./bin')
    print(type(d.r))
