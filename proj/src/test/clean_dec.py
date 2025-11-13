from tree_sitter import Language, Parser, Node
import tree_sitter_cpp
import r2pipe as r2
import re

remove_regex = [

    r"vector__",
    r"\.end__",
    r"__normal_iterator_",
    r"\.insert_",
    r"std::allocator",
    r"sym\.imp\.memset",
    r"sym\.imp\.atoi",
    r"sym\.imp\.htons",
    r"sym\.imp\.inet_addr",
    r"method\.std::vector.*?\._vector__",
    r"method\.std::vector.*?vector_.*?",
    r"//.*",
]


patterns = [re.compile(rx) for rx in remove_regex]


def print_tree(node, source_code, indent=0):
    print('  ' * indent +
          f"{node.type}: {source_code[node.start_byte:node.end_byte].decode('utf-8')}")
    for child in node.children:
        print_tree(child, source_code, indent + 1)


def parse(code):
    code = remove_comments(code)

    code = bytes(code, encoding='utf-8')

    CPP_LANGUAGE = Language(tree_sitter_cpp.language())
    parser = Parser(language=CPP_LANGUAGE)
    tree = parser.parse(code)
    root = tree.root_node

    print_tree(root, code)

    cuts = []

    for child in root.children:
        if child.type == 'function_definition':
            f_ev = eval_func(child, code[child.start_byte:child.end_byte])
            print(f'{repr(str(child.text))}\n\t{f_ev}')
            if f_ev:
                cuts.append((child.start_byte, child.end_byte))

    print(f'cuts: {cuts}')
    return carve(code, cuts)


def carve(data, cuts):
    mv = memoryview(data)
    return [mv[start:end].tobytes() for start, end in cuts]


def remove_comments(code: str) -> str:
    pattern = re.compile(
        r"""
        ("(?:\\.|[^"\\])*") |     # match double-quoted strings
        ('(?:\\.|[^'\\])*') |     # match single-quoted characters
        (//[^\n]*$)          |    # match single-line comments
        (/\*.*?\*/)               # match multi-line comments
        """,
        re.VERBOSE | re.MULTILINE | re.DOTALL,
    )

    def replacer(match):
        if match.group(1) or match.group(2):
            return match.group(0)
        return ''

    return re.sub(pattern, replacer, code)


def eval_func(func_node: Node, source: bytes):
    body = None
    for child in func_node.children:
        if child.type == 'compound_statement':
            body = child
            break
    if not body:
        return False

    # Artifact functions (only an unresolved indirect call)

    inst = [i for i in body.children if i.is_named and i.type != '}']
    if len(inst) != 2:
        return False
    call, ret = inst
    if call.type == 'expression_statement' and ret.type == 'return_statement':
        func = call.child_by_field_name('function')
        if not func:
            return False

        text = source[func_node.start_byte:func_node.end_byte]
        if re.match(rb'\(\*\*0x[0-9a-fA-F]+\)', text):
            return True

    return False


if __name__ == '__main__':
    r = r2.open(
        './juliet_bins/stack_buff_ofw/CWE121/bad/CWE121_Stack_Based_Buffer_Overflow__CWE135_04-bad', flags=['-2'])
    r.cmd('-AA')

    d = r.cmd('pdg@@@F') or ''

    s = parse(d)

#    print(d)
    print('-'*128)
    str(print(s))
