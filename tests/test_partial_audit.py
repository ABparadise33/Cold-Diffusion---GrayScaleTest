import importlib.util
import json
from pathlib import Path
import zipfile
import hashlib

from PIL import Image


def test_archive_audit_and_nonzero_chroma_inversion_without_network(tmp_path):
    spec = importlib.util.spec_from_file_location(
        'partial_audit', Path(__file__).resolve().parents[1] / 'tools/analyze_official_partial.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    raw_dir = tmp_path / 'raw'
    raw_dir.mkdir()
    for name in ('a', 'b'):
        Image.new('RGB', (17, 11), (180, 90, 30)).save(raw_dir / f'{name}.png')
    split = tmp_path / 'split.json'
    split.write_text(json.dumps({'test': ['a.png', 'b.png']}))
    archive = tmp_path / 'test.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        for percent, start in [(5, 19), (25, 15)]:
            prefix = f'paper_algorithm2/retain_{percent}pct/'
            scores = {'psnr': 30., 'ssim': .9, 'delta_e76': 5.}
            report = {**scores, 'monotonic': .5, 'baselines': {'raw': scores, 'input': scores},
                      'output_vs_raw': {'prediction_mae_255': 0.}, 'evaluation': {
                          'checkpoint_step': 50000, 'start_step': start,
                          'retained_raw_color_percent': percent, 'sampler': 'paper_algorithm2',
                          'split_sha256': hashlib.sha256(split.read_bytes()).hexdigest(),
                          'num_images': 2, 'spatial_mode': 'original_size', 'tile_size': 256,
                          'tile_overlap': 32, 'checkpoint_sha256': 'a' * 64}}
            rows = [{'image': name, 'raw': scores, 'prediction': scores} for name in ('a', 'b')]
            z.writestr(prefix + '其餘/metrics.json', json.dumps(report))
            z.writestr(prefix + '其餘/per_image_core.json', json.dumps(rows))
            for name in ('a', 'b'):
                z.write(raw_dir / f'{name}.png', prefix + f'predictions/{name}.png')
    result = module.audit(archive, raw_dir, split)
    assert result['num_images'] == 2
    assert result['conditions']['5']['png_mae_to_raw_255'] == 0
    assert result['conditions']['25']['delta_e_gain_ci95'] == [0., 0.]
    toy = result['analytic_control']['means_mae_255']['0.5']
    assert toy['float_mae'] < .01
    assert toy['quantized_mae'] > toy['float_mae']
