from __future__ import annotations

import copy
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from .bridge import GrayBridge
from .color import (
    denormalize_rgb,
    gray_anchor,
    normalize_rgb,
    normalized_lab_to_rgb,
    rgb_channel_mean_gray,
    rgb_to_normalized_lab,
)
from .io import append_csv, atomic_torch_save, save_stage_strip, save_trajectory_grid, update_ema
from .metrics import delta_e76, psnr, ssim, trajectory_monotonic_fraction
from .factory import ITERATIVE_MODES, NATURAL_MODES, RGB_MODES


NATURAL_IMAGE_MODES = NATURAL_MODES


def _restore_cuda_rng_states(states):
    """CUDA RNG restoration requires CPU uint8 tensors after checkpoint loading."""
    torch.cuda.set_rng_state_all([state.cpu() for state in states])


class Trainer:
    def __init__(self, model, bridge, train_loader, val_loader, config, device):
        self.model = model.to(device)
        self.ema = copy.deepcopy(model).to(device).eval().requires_grad_(False)
        self.bridge: GrayBridge = bridge.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.mode = config["mode"]
        self.color_space = "rgb" if self.mode in RGB_MODES else "lab"
        self.output = Path(config["output_dir"])
        self.output.mkdir(parents=True, exist_ok=True)
        train_cfg = config["training"]
        self.max_steps = int(train_cfg["max_steps"])
        self.grad_accum = int(train_cfg["grad_accum"])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(train_cfg["learning_rate"]))
        self.amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        except (AttributeError, TypeError):
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        self.step = 0
        self.best_psnr = -math.inf
        self._train_iter = iter(train_loader)

    def _next_batch(self):
        try:
            return next(self._train_iter)
        except StopIteration:
            self._train_iter = iter(self.train_loader)
            return next(self._train_iter)

    def _autocast(self):
        if self.amp:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    def _prepare(self, batch):
        raw_rgb = batch["raw"].to(self.device, non_blocking=True)
        reference_rgb = batch["reference"].to(self.device, non_blocking=True)
        if self.color_space == "rgb":
            raw_state = normalize_rgb(raw_rgb)
            reference_state = normalize_rgb(reference_rgb)
            # Keep the grayscale endpoint identical for the 1.0 and 1.5 runs;
            # only the color target changes between the two controls.
            anchor = rgb_channel_mean_gray(raw_state)
            return raw_rgb, reference_rgb, raw_state, reference_state, anchor
        raw_lab = rgb_to_normalized_lab(raw_rgb)
        reference_lab = rgb_to_normalized_lab(reference_rgb)
        return raw_rgb, reference_rgb, raw_lab, reference_lab, gray_anchor(raw_lab)

    def _state_to_rgb(self, state):
        if self.color_space == "rgb":
            return denormalize_rgb(state)
        return normalized_lab_to_rgb(state)

    def _state_to_lab(self, state):
        return rgb_to_normalized_lab(self._state_to_rgb(state))

    def _training_pair(self, raw_state, reference_state, anchor):
        batch = reference_state.shape[0]
        if self.mode in ITERATIVE_MODES:
            t = torch.randint(1, self.bridge.steps + 1, (batch,), device=self.device)
            model_input = self.bridge.degrade(reference_state, anchor, t)
        elif self.mode == "gray_oneshot":
            t = torch.full((batch,), self.bridge.steps, device=self.device, dtype=torch.long)
            model_input = anchor
        elif self.mode == "rgb_oneshot":
            t = torch.full((batch,), self.bridge.steps, device=self.device, dtype=torch.long)
            model_input = raw_state
        else:
            raise ValueError(f"unsupported mode: {self.mode}")
        return model_input, reference_state, t

    @torch.no_grad()
    def predict(self, model, raw_state, anchor, return_trajectory=False):
        batch = raw_state.shape[0]
        full_t = torch.full((batch,), self.bridge.steps, device=self.device, dtype=torch.long)
        if self.mode in ITERATIVE_MODES:
            return self.bridge.sample(model, anchor, return_trajectory=return_trajectory)
        model_input = anchor if self.mode == "gray_oneshot" else raw_state
        pred = model(model_input, full_t).clamp(-1, 1)
        return (pred, [model_input, pred]) if return_trajectory else pred

    def _checkpoint_payload(self):
        payload = {
            "step": self.step,
            "best_psnr": self.best_psnr,
            "model": self.model.state_dict(),
            "ema": self.ema.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "config": self.config,
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
            },
        }
        if torch.cuda.is_available():
            payload["rng"]["cuda"] = torch.cuda.get_rng_state_all()
        return payload

    def save_checkpoint(self, is_best=False):
        checkpoint_dir = self.output / "checkpoints"
        payload = self._checkpoint_payload()
        atomic_torch_save(payload, checkpoint_dir / "latest.pt")
        atomic_torch_save(payload, checkpoint_dir / f"step_{self.step:06d}.pt")
        if is_best:
            atomic_torch_save(payload, checkpoint_dir / "best.pt")

    def load_checkpoint(self, path: str | Path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        saved_config = checkpoint.get("config", {})
        for key in ("mode", "model", "diffusion"):
            if key in saved_config and saved_config[key] != self.config[key]:
                raise ValueError(f"resume config mismatch for {key}: {saved_config[key]!r} != {self.config[key]!r}")
        if self.mode in NATURAL_IMAGE_MODES:
            saved_factor = float(saved_config.get("data", {}).get("saturation_factor", 1.0))
            current_factor = float(self.config.get("data", {}).get("saturation_factor", 1.0))
            if saved_factor != current_factor:
                raise ValueError(
                    "resume saturation mismatch: "
                    f"{saved_factor:g} != {current_factor:g}"
                )
        saved_reference_factor = float(
            saved_config.get("data", {}).get("reference_saturation_factor", 1.0)
        )
        current_reference_factor = float(
            self.config.get("data", {}).get("reference_saturation_factor", 1.0)
        )
        if saved_reference_factor != current_reference_factor:
            raise ValueError(
                "resume reference saturation mismatch: "
                f"{saved_reference_factor:g} != {current_reference_factor:g}"
            )
        self.model.load_state_dict(checkpoint["model"])
        self.ema.load_state_dict(checkpoint["ema"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scaler"):
            self.scaler.load_state_dict(checkpoint["scaler"])
        self.step = int(checkpoint["step"])
        self.best_psnr = float(checkpoint.get("best_psnr", -math.inf))
        rng = checkpoint.get("rng", {})
        if "python" in rng:
            random.setstate(rng["python"])
        if "numpy" in rng:
            np.random.set_state(rng["numpy"])
        if "torch" in rng:
            torch.set_rng_state(rng["torch"].cpu())
        if self.device.type == "cuda" and "cuda" in rng:
            # map_location=self.device also maps saved RNG tensors to CUDA,
            # while set_rng_state_all requires CPU torch.ByteTensor values.
            _restore_cuda_rng_states(rng["cuda"])
        print(f"resumed from step {self.step}: {path}")

    @torch.no_grad()
    def validate(self):
        self.ema.eval()
        totals = {"psnr": 0.0, "ssim": 0.0, "delta_e76": 0.0, "monotonic": 0.0}
        count = 0
        sample = None
        preview_name = str(
            self.config.get("data", {}).get("validation_preview_name", "")
        )
        max_batches = int(self.config["training"].get("max_val_batches", 20))
        for batch_index, batch in enumerate(self.val_loader):
            if batch_index >= max_batches:
                break
            raw_rgb, reference_rgb, raw_state, reference_state, anchor = self._prepare(batch)
            pred_state, trajectory = self.predict(
                self.ema, raw_state, anchor, return_trajectory=True
            )
            pred_rgb = self._state_to_rgb(pred_state)
            pred_lab = rgb_to_normalized_lab(pred_rgb)
            reference_lab = rgb_to_normalized_lab(reference_rgb)
            trajectory_lab = [self._state_to_lab(state) for state in trajectory]
            batch_size = pred_rgb.shape[0]
            totals["psnr"] += psnr(pred_rgb, reference_rgb).sum().item()
            totals["ssim"] += ssim(pred_rgb, reference_rgb).sum().item()
            totals["delta_e76"] += delta_e76(pred_lab, reference_lab).sum().item()
            totals["monotonic"] += trajectory_monotonic_fraction(
                trajectory_lab, reference_lab
            ).sum().item()
            count += batch_size
            batch_names = [str(name) for name in batch["name"]]
            should_capture = sample is None and (
                not preview_name or preview_name in batch_names
            )
            if should_capture:
                image_index = batch_names.index(preview_name) if preview_name else 0
                sample_slice = slice(image_index, image_index + 1)
                full_t = torch.full((batch_size,), self.bridge.steps, device=self.device, dtype=torch.long)
                direct_state = self.ema(anchor, full_t).clamp(-1, 1)
                sample = (
                    raw_rgb[sample_slice],
                    self._state_to_rgb(anchor)[sample_slice],
                    self._state_to_rgb(direct_state)[sample_slice],
                    pred_rgb[sample_slice],
                    reference_rgb[sample_slice],
                    [state[sample_slice] for state in trajectory],
                )
        if count == 0:
            raise RuntimeError("validation loader produced no samples")
        metrics = {key: value / count for key, value in totals.items()}
        if sample is not None:
            raw, gray, direct, pred, reference, trajectory = sample
            save_stage_strip(
                [("raw", raw), ("gray anchor", gray), ("direct", direct), ("sampled", pred), ("reference", reference)],
                self.output / "samples" / f"step_{self.step:06d}.png",
                display_scale=4,
            )
            save_trajectory_grid(
                trajectory,
                self.output / "trajectories" / f"step_{self.step:06d}.png",
                display_scale=4,
                color_space=self.color_space,
            )
        return metrics

    def step_ema(self):
        cfg = self.config["training"]
        if self.step % int(cfg["ema_update_every"]) == 0:
            update_ema(self.ema, self.model, float(cfg["ema_decay"]))

    def train(self):
        train_cfg = self.config["training"]
        log_every = int(train_cfg["log_every"])
        validate_every = int(train_cfg["validate_every"])
        save_every = int(train_cfg["save_every"])
        running = []
        start = time.time()
        self.optimizer.zero_grad(set_to_none=True)

        while self.step < self.max_steps:
            step_losses = []
            for _ in range(self.grad_accum):
                batch = self._next_batch()
                _, _, raw_state, reference_state, anchor = self._prepare(batch)
                model_input, target, t = self._training_pair(
                    raw_state, reference_state, anchor
                )
                with self._autocast():
                    prediction = self.model(model_input, t)
                    loss = (prediction - target).abs().mean()
                self.scaler.scale(loss / self.grad_accum).backward()
                step_losses.append(loss.detach().item())
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            self.step += 1
            self.step_ema()
            running.append(sum(step_losses) / len(step_losses))

            if self.step % log_every == 0:
                elapsed = time.time() - start
                avg_loss = sum(running) / len(running)
                steps_per_second = len(running) / max(elapsed, 1e-9)
                eta_hours = (self.max_steps - self.step) / max(steps_per_second, 1e-9) / 3600
                gpu_memory = ""
                if self.device.type == "cuda":
                    peak_gb = torch.cuda.max_memory_allocated(self.device) / 1024**3
                    gpu_memory = f" peak_vram={peak_gb:.2f}GB"
                print(
                    f"step={self.step} loss={avg_loss:.6f} "
                    f"speed={steps_per_second:.2f}step/s eta={eta_hours:.2f}h{gpu_memory}"
                )
                append_csv(self.output / "metrics.csv", {
                    "step": self.step,
                    "train_l1": avg_loss,
                    "val_psnr": "",
                    "val_ssim": "",
                    "val_delta_e76": "",
                    "trajectory_monotonic": "",
                })
                running.clear()
                start = time.time()

            validation = None
            if self.step % validate_every == 0 or self.step == self.max_steps:
                validation = self.validate()
                print("validation", {k: round(v, 5) for k, v in validation.items()})
                append_csv(self.output / "metrics.csv", {
                    "step": self.step,
                    "train_l1": "",
                    "val_psnr": validation["psnr"],
                    "val_ssim": validation["ssim"],
                    "val_delta_e76": validation["delta_e76"],
                    "trajectory_monotonic": validation["monotonic"],
                })

            if self.step % save_every == 0 or self.step == self.max_steps:
                score = validation["psnr"] if validation is not None else -math.inf
                is_best = score > self.best_psnr
                if is_best:
                    self.best_psnr = score
                self.save_checkpoint(is_best=is_best)
