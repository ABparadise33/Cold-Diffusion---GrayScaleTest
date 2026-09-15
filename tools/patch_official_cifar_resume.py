"""Patch only official checkpoint plumbing, counting, and save cadence."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil


def patch(root):
    root = Path(root)
    source = root/'diffusion/diffusion.py'
    train_path = root/'train.py'
    text = source.read_text()
    training = train_path.read_text()
    support = Path(__file__).with_name('official_cifar_resume_support.py')
    marker = '# CODEX_CIFAR_RESUME_V1'
    if marker not in text:
        parsed = ast.parse(text)
        trainer = next(n for n in parsed.body if isinstance(n, ast.ClassDef) and n.name == 'Trainer')
        methods = {n.name: n for n in trainer.body if isinstance(n, ast.FunctionDef)}
        lines = text.splitlines(keepends=True)
        replacements = {
            'save': '    def save(self, save_with_time_stamp=False):\n        return _cifar_save(self, save_with_time_stamp)\n',
            'load': '    def load(self, load_path):\n        return _cifar_load(self, load_path)\n',
        }
        for name in sorted(replacements, key=lambda k: methods[k].lineno, reverse=True):
            node = methods[name]
            lines[node.lineno-1:node.end_lineno] = [replacements[name]]
        updated = ''.join(lines)
        start = updated.index('    def train(self):')
        end = updated.index('    def save_gif(', start)
        block = updated[start:end]
        assert block.count('            self.step += 1') == 1
        block = block.replace('            self.step += 1', '')
        needle = '            if self.step != 0 and self.step % self.save_and_sample_every == 0:'
        assert block.count(needle) == 1
        # EMA still uses the upstream zero-based counter. Save/report after increment.
        block = block.replace(needle, '            self.step += 1\n\n' + needle)
        block = block.replace("        print('training completed')", "        self.save()\n        self.save(save_with_time_stamp=True)\n        print('training completed')")
        updated = marker+'\nfrom codex_cifar_resume import save_checkpoint as _cifar_save, load_checkpoint as _cifar_load\n'+updated[:start]+block+updated[end:]
        ast.parse(updated)
        backup = source.with_name('diffusion.py.before_cifar_resume')
        if backup.exists():
            raise FileExistsError(f'backup already exists: {backup}')
        shutil.copy2(source, backup)
        source.write_text(updated)
    # The preceding pilot instructions may have supplied explicit 1k cadence.
    # Preserve other Trainer arguments; set checkpoint milestones to10k to bound disk use.
    tree = ast.parse(training)
    call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'Trainer')
    rows = training.splitlines(keepends=True)
    edits = []
    present = set()
    for kw in call.keywords:
        if kw.arg in ('save_and_sample_every', 'save_with_time_stamp_every'):
            value = 1000 if kw.arg == 'save_and_sample_every' else 10000
            present.add(kw.arg)
            line = rows[kw.value.lineno-1]
            edits.append((kw.value.lineno-1, line[:kw.value.col_offset]+str(value)+line[kw.value.end_col_offset:]))
    for index, line in sorted(edits, reverse=True):
        rows[index] = line
    training = ''.join(rows)
    # Missing options go after the positional arguments, before first keyword.
    missing = []
    for key, value in [('save_and_sample_every',1000),('save_with_time_stamp_every',10000)]:
        if key not in present:
            missing.append(f'    {key}={value},\n')
    if missing:
        pos = training.index('    image_size = image_size,', training.index('trainer = Trainer('))
        training = training[:pos]+''.join(missing)+training[pos:]
    ast.parse(training)
    # Avoid duplicated final saves from the previous manual patch.
    training = training.replace('trainer.train()\ntrainer.save()\ntrainer.save(save_with_time_stamp=True)', 'trainer.train()')
    train_path.write_text(training)
    shutil.copy2(support, root/'codex_cifar_resume.py')
    metadata = {'version':1,'changes':['checkpoint optimizer/RNG', 'completed update counters after EMA', 'final save', '1k latest/preview and10k milestones'],
                'data_loader': 'new iterator on resume, not bitwise data-order replay',
                'sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,train_path,root/'codex_cifar_resume.py']}}
    (root/'cifar_resume_patch.json').write_text(json.dumps(metadata,indent=2))
    print('Official resume patch ready:', root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--official-dir', required=True)
    patch(parser.parse_args().official_dir)
