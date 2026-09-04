"""Explicit model/operator routing; historical modes are never reinterpreted."""
from .bridge import GrayBridge
from .model import RestorationUNet

OFFICIAL_MODE = "official_rgb_colorization"
RGB_MODES = {"natural_rgb_colorization", OFFICIAL_MODE}
NATURAL_MODES = RGB_MODES | {"natural_lab_colorization"}
ITERATIVE_MODES = NATURAL_MODES | {"cold_gray"}


def build_model_and_bridge(config):
    cfg = config["model"]
    steps = int(config["diffusion"]["steps"])
    if config["mode"] == OFFICIAL_MODE:
        from .official_colorization import OfficialColorizer, RGBDecolorization
        if float(config["data"].get("saturation_factor", 1)) != 1.0:
            raise ValueError("official baseline requires saturation_factor=1.0")
        if cfg.get("architecture") != "upstream_convnext":
            raise ValueError("official baseline requires upstream_convnext")
        model = OfficialColorizer(int(cfg["dim"]), tuple(cfg["dim_mults"]), steps)
        bridge = RGBDecolorization(steps, config["diffusion"]["sampler"])
        return model, bridge
    if "architecture" in cfg:
        raise ValueError("explicit architecture is supported only in official_rgb_colorization")
    return RestorationUNet(
        int(cfg["base_channels"]), tuple(cfg["channel_mults"]),
        float(cfg.get("dropout", 0)), steps,
    ), GrayBridge(steps)
