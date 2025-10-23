from pathlib import Path
import r2pipe as r2
import argparse
import os
from tqdm import tqdm
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import decompile


def decompile_label(bin: Path):
    dec = decompile.decompile_filt_f(str(bin.resolve()))
    filename = str(bin.resolve().name)
    if filename[-3:] == 'bad':
        label = 1
    else:
        label = 0
    entry = {'code': dec, 'label': label, 'file': filename}
    return entry


def make_jsonl_db(dir, out, thread_num=4):
    cases = []
    # for cwe in tqdm(Path(dir).glob('*/'), desc='ok', position=0, ascii=True):
    cases.extend(list(dir.resolve().glob('*[good bad]/*')))

    with open(f'{out.resolve()}/db.jsonl', 'w') as out, ThreadPoolExecutor(max_workers=thread_num) as exec:
        jobs = {exec.submit(decompile_label, bin): bin for bin in cases}
        for job in tqdm(as_completed(jobs), total=len(jobs), desc='Decompiling', ascii=True):
            res = job.result()
            out.write(json.dumps(res) + '\n')
            out.flush()
            tqdm.write(f"[-] {res['file']} done - {res['label']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source', default='./bin/',
                        action='store', type=str)
    parser.add_argument('-o', '--out', default='./db/',
                        action='store', type=str)

    args = parser.parse_args()

    dir = Path(args.out)
    src = Path(args.source)
    if not Path.exists(dir):
        os.mkdir(dir)

    make_jsonl_db(src, dir, 8)
