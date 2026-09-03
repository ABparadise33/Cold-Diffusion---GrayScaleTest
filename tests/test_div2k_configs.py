from pathlib import Path

import yaml


CONFIGS = {
    1.0: "div2k_rgb_sat1_50k.yaml",
    1.25: "div2k_rgb_sat1_25_50k.yaml",
    1.5: "div2k_rgb_sat1_5_50k.yaml",
    2.0: "div2k_rgb_sat2_50k.yaml",
}


def test_div2k_saturation_sweep_changes_only_target_factor_and_output():
    root = Path(__file__).resolve().parents[1] / "configs"
    loaded = {
        factor: yaml.safe_load((root / filename).read_text(encoding="utf-8"))
        for factor, filename in CONFIGS.items()
    }
    baseline = loaded[1.0]
    outputs = set()

    for factor, config in loaded.items():
        assert config["mode"] == "natural_rgb_colorization"
        assert float(config["data"]["saturation_factor"]) == factor
        assert config["seed"] == baseline["seed"] == 42
        assert config["data"]["image_size"] == baseline["data"]["image_size"]
        assert config["model"] == baseline["model"]
        assert config["diffusion"] == baseline["diffusion"]
        assert config["training"] == baseline["training"]
        outputs.add(config["output_dir"])

    assert len(outputs) == len(CONFIGS)


def test_div2k_lab_sat1_changes_only_color_space_mode_and_output():
    root = Path(__file__).resolve().parents[1] / "configs"
    rgb = yaml.safe_load((root / "div2k_rgb_sat1_50k.yaml").read_text(encoding="utf-8"))
    lab = yaml.safe_load((root / "div2k_lab_sat1_50k.yaml").read_text(encoding="utf-8"))

    assert rgb["mode"] == "natural_rgb_colorization"
    assert lab["mode"] == "natural_lab_colorization"
    assert lab["experiment"] == "div2k_lab_sat1_50k"
    assert lab["output_dir"] == "outputs/div2k_lab_sat1_50k"

    for config in (rgb, lab):
        config.pop("experiment")
        config.pop("mode")
        config.pop("output_dir")
    assert lab == rgb
