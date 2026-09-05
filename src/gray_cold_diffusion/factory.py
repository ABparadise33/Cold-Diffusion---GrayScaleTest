"""Explicit model/operator routing; historical modes are never reinterpreted."""
from .bridge import GrayBridge
from .model import RestorationUNet

OFFICIAL_MODE = "official_rgb_colorization"
SPATIAL_MODE = "spatial_chroma_colorization"
UPSTREAM_MODES = {OFFICIAL_MODE, SPATIAL_MODE}
RGB_MODES = {"natural_rgb_colorization", *UPSTREAM_MODES}
NATURAL_MODES = RGB_MODES | {"natural_lab_colorization"}
ITERATIVE_MODES = NATURAL_MODES | {"cold_gray"}


def build_model_and_bridge(config):
    cfg = config["model"]
    steps = int(config["diffusion"]["steps"])
    if config["mode"] in UPSTREAM_MODES:
        from .official_colorization import OfficialColorizer, RGBDecolorization
        if float(config["data"].get("saturation_factor", 1)) != 1.0:
            raise ValueError("upstream-model experiments require saturation_factor=1.0")
        if cfg.get("architecture") != "upstream_convnext":
            raise ValueError("upstream-model experiments require upstream_convnext")
        model = OfficialColorizer(int(cfg["dim"]), tuple(cfg["dim_mults"]), steps)
        if config["mode"] == OFFICIAL_MODE:
            bridge = RGBDecolorization(steps, config["diffusion"]["sampler"])
        else:
            from .spatial_chroma import SpatialChromaMask
            bridge = SpatialChromaMask(
                steps, config["diffusion"]["sampler"],
                int(config["diffusion"].get("sampling_seed", config["seed"])),
            )
        return model, bridge
    if "architecture" in cfg:
        raise ValueError("explicit architecture is supported only in upstream-model modes")
    return RestorationUNet(
        int(cfg["base_channels"]), tuple(cfg["channel_mults"]),
        float(cfg.get("dropout", 0)), steps,
    ), GrayBridge(steps)
