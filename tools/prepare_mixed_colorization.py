"""Create a namespaced, split-preserving UIEB-reference + DIV2K-HR dataset."""
import argparse
import hashlib
import json
from pathlib import Path


def prepare(uieb_reference, split_file, div2k_train, div2k_val, output):
    split = json.loads(Path(split_file).read_text())
    def references(label):
        return [Path(uieb_reference) / (x if isinstance(x, str) else x['reference'])
                for x in split[label]]
    def images(directory):
        return sorted(Path(directory).glob('*.png'))
    pools = {'train': {'UIEB': references('train'), 'DIV2K': images(div2k_train)},
             'val': {'UIEB': references('val'), 'DIV2K': images(div2k_val)}}
    test = {p.resolve() for p in references('test')} if 'test' in split else set()
    resolved = {}
    hashes = {}
    records = {}
    for label, domains in pools.items():
        resolved[label], hashes[label], records[label] = set(), set(), []
        for domain, paths in domains.items():
            if not paths:
                raise ValueError(f'empty {label}/{domain}')
            for path in paths:
                if not path.is_file():
                    raise FileNotFoundError(path)
                source = path.resolve()
                if source in resolved[label]:
                    raise ValueError(f'duplicate source: {source}')
                resolved[label].add(source)
                digest = hashlib.sha256(source.read_bytes()).hexdigest()
                hashes[label].add(digest)
                records[label].append({'domain': domain, 'source': str(source),
                                       'name': f'{domain}__{path.name}', 'sha256': digest})
    if resolved['train'] & resolved['val'] or hashes['train'] & hashes['val']:
        raise ValueError('train/val leakage (path or exact file content)')
    if (resolved['train'] | resolved['val']) & test:
        raise ValueError('UIEB test split leakage')
    test_hashes = {hashlib.sha256(p.read_bytes()).hexdigest() for p in test}
    if (hashes['train'] | hashes['val']) & test_hashes:
        raise ValueError('UIEB test content leakage')
    manifest = {'task': 'synthetic full-gray colorization; UIEB references, never raw inputs',
                'split_sha256': hashlib.sha256(Path(split_file).read_bytes()).hexdigest(),
                'records': records}
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        if not (output/'manifest.json').exists() or json.loads((output/'manifest.json').read_text()) != manifest:
            raise FileExistsError('mixed dataset exists with a different or incomplete manifest')
        for label, rows in records.items():
            if {p.name for p in (output/label).iterdir()} != {r['name'] for r in rows}:
                raise ValueError('mixed directory listing changed')
            for row in rows:
                if (output/label/row['name']).resolve() != Path(row['source']):
                    raise ValueError('mixed symlink changed')
        return manifest
    for label, rows in records.items():
        (output/label).mkdir(parents=True, exist_ok=True)
        for row in rows:
            (output/label/row['name']).symlink_to(row['source'])
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    return manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--uieb-reference', required=True)
    p.add_argument('--split-file', default='splits/uieb_seed42.json')
    p.add_argument('--div2k-train', required=True)
    p.add_argument('--div2k-val', required=True)
    p.add_argument('--output', default='data/UIEB_DIV2K')
    args = p.parse_args()
    manifest = prepare(**vars(args))
    print({label: len(rows) for label, rows in manifest['records'].items()})


if __name__ == '__main__':
    main()
