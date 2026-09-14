"""Select pinned PyTorch wheels from detected GPU capabilities, before torch exists."""
import argparse
import json
from pathlib import Path
import re
import subprocess


STACKS = {
    'cu121': ('2.5.1', '0.20.1', 'cu121', '12.1'),
    'cu128': ('2.7.1', '0.22.1', 'cu128', '12.8'),
}


def choose_stack(capabilities):
    capabilities = set(capabilities)
    if not capabilities or not capabilities <= {'8.6', '8.9', '12.0'}:
        raise ValueError(f'Unconfigured GPU capabilities: {sorted(capabilities)}; inspect GPU before selecting wheels')
    return STACKS['cu128' if '12.0' in capabilities else 'cu121']


def detect_capabilities():
    query = subprocess.run(
        ['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'],
        capture_output=True, text=True, check=False,
    )
    if query.returncode == 0:
        values = [line.strip() for line in query.stdout.splitlines() if line.strip()]
        if values and all(re.fullmatch(r'\d+\.\d+', value) for value in values):
            return values, 'nvidia-smi compute_cap'
    # Older nvidia-smi releases may not expose compute_cap. Match exact known
    # model numbers; never default an unrecognized GPU to a 4090/legacy stack.
    names = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], text=True,
    ).splitlines()
    mapping = {'3090': '8.6', '4090': '8.9', '5090': '12.0'}
    capabilities = []
    for name in names:
        match = re.search(r'\bRTX\s+(3090|4090|5090)(?:\s+Ti)?\s*$', name.strip())
        if not match:
            raise ValueError(f'Cannot determine compute capability for {name!r}')
        capabilities.append(mapping[match.group(1)])
    return capabilities, 'exact RTX model fallback'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--record', type=Path)
    args = parser.parse_args()
    capabilities, detection = detect_capabilities()
    stack = choose_stack(capabilities)
    if args.record:
        args.record.write_text(json.dumps({
            'capabilities': capabilities, 'detection': detection,
            'torch': stack[0], 'torchvision': stack[1], 'wheel_index': stack[2],
            'cuda_runtime': stack[3],
            'scope': 'all GPUs reported by nvidia-smi; use newest required stack',
        }, indent=2))
    print(' '.join(stack))


if __name__ == '__main__':
    main()
