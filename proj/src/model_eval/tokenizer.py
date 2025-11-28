from tokenizers import processors, pre_tokenizers
from tokenizers import normalizers, decoders
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.processors import TemplateProcessing
from tokenizers.normalizers import StripAccents, unicode_normalizer_from_str, Replace
from clang import *
from clang import cindex
from tokenizers.pre_tokenizers import PreTokenizer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers import NormalizedString, PreTokenizedString
from typing import List

if cindex.Config.library_file is None:
    cindex.Config.set_library_file('/usr/lib/llvm-15/lib/libclang.so')


class ClangTokenizer:
    cidx = cindex.Index.create()

    def clang_split(self, i: int, normalized_string: NormalizedString) -> List[NormalizedString]:
        # Tokkenize using clang
        tok = []
        tu = self.cidx.parse('tmp.c',
                             args=[''],
                             unsaved_files=[
                                 ('tmp.c', str(normalized_string.original))],
                             options=0)
        for t in tu.get_tokens(extent=tu.cursor.extent):
            spelling = t.spelling.strip()

            if spelling == '':
                continue

            tok.append(NormalizedString(spelling))

        return (tok)

    def pre_tokenize(self, pretok: PreTokenizedString):
        pretok.split(self.clang_split)


def load_tokenizer():
    vocab, merges = BPE.read_file(
        vocab="./models/tokenizer/drapgh-vocab.json", merges="./models/tokenizer/drapgh-merges.txt")
    tokenizer = Tokenizer(
        BPE(vocab, merges, unk_token="<unk>"))

    tokenizer.normalizer = normalizers.Sequence(
        [StripAccents(), Replace(" ", "Ä")])
    tokenizer.pre_tokenizer = PreTokenizer.custom(ClangTokenizer())
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    tokenizer.post_processor = TemplateProcessing(
        single="<s> $A </s>",
        special_tokens=[
            ("<s>", 0),
            ("<pad>", 1),
            ("</s>", 2),
            ("<unk>", 3),
            ("<mask>", 4)
        ]
    )

    tokenizer.enable_truncation(max_length=1024)
    tokenizer.enable_padding(direction='right', pad_id=1, pad_type_id=0,
                             pad_token='<pad>', length=None, pad_to_multiple_of=None)
    return tokenizer


if __name__ == '__main__':
    t = load_tokenizer()
    print(type(t))
