import decompile


if __name__ == '__main__':
    d = decompile.Decompiler('./bin')
    f = d.enum_f()
    s = d.plt_f_sym(f)

    print()
