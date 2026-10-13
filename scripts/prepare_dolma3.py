"""Download public Dolma3 and build Qwen2-tokenized Megatron datasets."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
import hashlib
import io
import json
import multiprocessing
import os
from pathlib import Path
import sys

DATASET = 'allenai/dolma3_mix-150B-1025'
REVISION = 'afa92bfb22366821c5e6cd427cdd036b34b713ef'
TOKENIZER = 'Qwen/Qwen2-0.5B'
TOKENIZER_REVISION = '91d2aff3f957f99e4c74c962f2f408dcc88a18d8'
TOKENIZER_SHA256 = 'f7c9b2dba4a296b1aa76c16a34b8225c0c118978400d4bb66bff0902d702f5b8'
SPLITS = ('train', 'valid', 'test')


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def prepare_group(task):
    index, files, output, tokenizer_path, fingerprint = task
    import numpy as np
    import torch
    import zstandard
    from tokenizers import Tokenizer
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Megatron-LM'))
    from megatron.core.datasets.indexed_dataset import IndexedDatasetBuilder
    torch.set_num_threads(1)
    directory = output / 'indexed' / f'{index:05d}'
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / 'complete.json'
    if marker.exists():
        record = json.loads(marker.read_text())
        if record['fingerprint'] != fingerprint:
            raise ValueError(f'Different preparation settings in {directory}; use a new output directory')
        for name, size in record['sizes'].items():
            if (directory / name).stat().st_size != size:
                raise ValueError(f'Incomplete output: {directory / name}')
        return record
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    eos = tokenizer.token_to_id('<|endoftext|>')
    assert eos == 151643
    builders = {s: IndexedDatasetBuilder(str(directory / f'{s}.bin.partial'), dtype=np.int32) for s in SPLITS}
    tokens, documents = dict.fromkeys(SPLITS, 0), dict.fromkeys(SPLITS, 0)

    def consume(rows):
        encoded = tokenizer.encode_batch([row['text'] for row in rows], add_special_tokens=False)
        for row, encoding in zip(rows, encoded):
            ids = encoding.ids
            if not ids:
                continue
            key = row.get('id') or row['text']
            bucket = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'big') % 1000
            split = 'valid' if bucket == 0 else 'test' if bucket == 1 else 'train'
            ids.append(eos)
            builders[split].add_document(ids, [len(ids)])
            tokens[split] += len(ids)
            documents[split] += 1

    for path in files:
        with path.open('rb') as raw, zstandard.ZstdDecompressor().stream_reader(raw) as stream:
            with io.TextIOWrapper(stream, encoding='utf-8') as lines:
                batch = []
                for line in lines:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if not isinstance(row['text'], str):
                        raise ValueError(f'Invalid text field in {path}')
                    batch.append(row)
                    if len(batch) == 256:
                        consume(batch)
                        batch = []
                if batch:
                    consume(batch)
    for split, builder in builders.items():
        builder.finalize(str(directory / f'{split}.idx.partial'))
        for suffix in ('bin', 'idx'):
            (directory / f'{split}.{suffix}.partial').replace(directory / f'{split}.{suffix}')
    result = {'fingerprint': fingerprint, 'tokens': tokens, 'documents': documents,
              'prefixes': {s: str(directory / s) for s in SPLITS},
              'sizes': {p.name: p.stat().st_size for p in directory.iterdir() if p.suffix in ('.bin', '.idx')}}
    write_json(marker, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--download-workers', type=int, default=8)
    parser.add_argument('--groups', type=int, default=128)
    parser.add_argument('--skip-download', action='store_true', help='Use already downloaded raw/ and tokenizer/ directories')
    args = parser.parse_args()
    if min(args.workers, args.download_workers, args.groups) < 1:
        parser.error('Worker and group counts must be positive')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / '.prepare.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.environ.update(OMP_NUM_THREADS='1', RAYON_NUM_THREADS='2', TOKENIZERS_PARALLELISM='true')
    if not args.skip_download:
        from huggingface_hub import snapshot_download
        snapshot_download(DATASET, repo_type='dataset', revision=REVISION,
                          allow_patterns=['data/**/*.jsonl.zst', 'README.md'],
                          local_dir=str(output / 'raw'), max_workers=args.download_workers)
        snapshot_download(TOKENIZER, revision=TOKENIZER_REVISION,
                          allow_patterns=['tokenizer*', 'vocab.json', 'merges.txt', 'config.json', 'LICENSE'],
                          local_dir=str(output / 'tokenizer'), max_workers=args.download_workers)
    tokenizer_path = output / 'tokenizer/tokenizer.json'
    if hashlib.sha256(tokenizer_path.read_bytes()).hexdigest() != TOKENIZER_SHA256:
        raise ValueError('Tokenizer differs from the pinned public Qwen2 tokenizer')
    files = sorted((output / 'raw/data').rglob('*.jsonl.zst'))
    if not files:
        raise ValueError('No Dolma3 JSONL.zst files found')
    groups = [[] for _ in range(min(args.groups, len(files)))]
    sizes = [0] * len(groups)
    for path in sorted(files, key=lambda p: (-p.stat().st_size, str(p))):
        index = min(range(len(groups)), key=lambda i: (sizes[i], i))
        groups[index].append(path)
        sizes[index] += path.stat().st_size
    manifest = {'dataset': DATASET, 'revision': REVISION, 'tokenizer': TOKENIZER,
                'tokenizer_revision': TOKENIZER_REVISION, 'tokenizer_sha256': TOKENIZER_SHA256,
                'eos_id': 151643, 'split': 'sha256(id or text)[:8], big endian, modulo 1000: valid=0, test=1, train=others',
                'groups': [[{'path': str(p.relative_to(output / 'raw')), 'bytes': p.stat().st_size}
                            for p in sorted(group)] for group in groups]}
    fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    previous = output / 'source_manifest.json'
    if previous.exists() and json.loads(previous.read_text()) != manifest:
        raise ValueError('Input files or settings changed; use a new output directory')
    write_json(previous, manifest)
    tasks = [(i, sorted(group), output, tokenizer_path, fingerprint) for i, group in enumerate(groups)]
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        records = []
        for record in pool.map(prepare_group, tasks):
            records.append(record)
            print(f'Indexed {len(records)}/{len(tasks)} groups', flush=True)
    mixture = {s: [] for s in SPLITS}
    for record in records:
        for split in SPLITS:
            if record['tokens'][split]:
                mixture[split].extend([str(record['tokens'][split]), record['prefixes'][split]])
    if not all(mixture.values()):
        raise ValueError('A split is empty; prepare more documents before training')
    write_json(output / 'data_args.json', mixture)
    write_json(output / 'READY.json', {s: sum(r['tokens'][s] for r in records) for s in SPLITS})
    print(f'DATA_ARGS={output / "data_args.json"}\nTOKENIZER_DIR={output / "tokenizer"}')


if __name__ == '__main__':
    main()
