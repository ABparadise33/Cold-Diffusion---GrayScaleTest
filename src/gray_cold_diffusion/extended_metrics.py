"""FlowIE-compatible full-reference and no-reference underwater metrics."""

from __future__ import annotations

import csv
import math
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
import torch
from torch import nn


FR_METRICS = ("psnr_y", "psnr_rgb", "ssim_y", "ssim_rgb", "ms_ssim", "lpips", "mi", "kl")
NR_METRICS = ("niqe", "musiq", "clipiqa", "entropy", "uiqm", "ducd")
ALL_METRICS = FR_METRICS + NR_METRICS
COLD_METRICS = ("delta_e76", "trajectory_monotonic")
OUTPUT_METRICS = ALL_METRICS + COLD_METRICS
METRIC_DIRECTIONS = {
    "psnr_y": "higher",
    "psnr_rgb": "higher",
    "ssim_y": "higher",
    "ssim_rgb": "higher",
    "ms_ssim": "higher",
    "lpips": "lower",
    "niqe": "lower",
    "musiq": "higher",
    "clipiqa": "higher",
    "entropy": "higher",
    "mi": "higher",
    "kl": "lower",
    "uiqm": "higher",
    "ducd": "lower / legacy target about 1.8-2.0",
    "delta_e76": "lower",
    "trajectory_monotonic": "higher",
}


def differentiable_histogram2(values, bins=256, min_val=0.0, max_val=1.0, sigma=0.01):
    values = values.view(values.shape[0], values.shape[1], -1)
    centers = torch.linspace(min_val, max_val, bins, device=values.device).view(1, 1, 1, bins)
    weights = torch.exp(-0.5 * ((values.unsqueeze(-1) - centers) / sigma) ** 2)
    histogram = weights.sum(dim=2)
    return histogram / (histogram.sum(dim=-1, keepdim=True) + 1e-6)


def compute_corr(first, second):
    first = first - first.mean(dim=1, keepdim=True)
    second = second - second.mean(dim=1, keepdim=True)
    return (first * second).mean(dim=1) / (first.std(dim=1) * second.std(dim=1) + 1e-6)


class DUCD(nn.Module):
    def forward(self, image):
        red = differentiable_histogram2(image[:, 0:1])[0, 0].unsqueeze(0)
        green = differentiable_histogram2(image[:, 1:2])[0, 0].unsqueeze(0)
        blue = differentiable_histogram2(image[:, 2:3])[0, 0].unsqueeze(0)
        return torch.abs(compute_corr(red, green)) + torch.abs(compute_corr(green, blue)) + torch.abs(
            compute_corr(red, blue)
        )


def entropy(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    histogram, _ = np.histogram(gray.flatten(), 256, (0, 255), density=True)
    histogram = histogram[histogram > 0]
    return float(-np.sum(histogram * np.log2(histogram)))


def mutual_information(first, second):
    first = cv2.cvtColor(first, cv2.COLOR_RGB2GRAY)
    second = cv2.cvtColor(second, cv2.COLOR_RGB2GRAY)
    histogram, _, _ = np.histogram2d(first.flatten(), second.flatten(), bins=256)
    joint = histogram / np.sum(histogram)
    marginal_first = np.sum(joint, axis=1)
    marginal_second = np.sum(joint, axis=0)
    independent = marginal_first[:, None] * marginal_second[None, :]
    nonzero = joint > 0
    return float(np.sum(joint[nonzero] * np.log2(joint[nonzero] / independent[nonzero])))


def kl_divergence(first, second):
    first = cv2.cvtColor(first, cv2.COLOR_RGB2GRAY)
    second = cv2.cvtColor(second, cv2.COLOR_RGB2GRAY)
    histogram_first, _ = np.histogram(first.flatten(), 256, (0, 255), density=True)
    histogram_second, _ = np.histogram(second.flatten(), 256, (0, 255), density=True)
    histogram_first += 1e-10
    histogram_second += 1e-10
    return float(np.sum(histogram_first * np.log2(histogram_first / histogram_second)))


def _trimmed_mean(values, alpha_left=0.1, alpha_right=0.1):
    values = sorted(values)
    count = len(values)
    trim_left = math.ceil(alpha_left * count)
    trim_right = math.floor(alpha_right * count)
    weight = 1 / (count - trim_left - trim_right)
    return weight * sum(values[int(trim_left + 1) : int(count - trim_right)])


def _variance(values, mean):
    return sum(math.pow((pixel - mean), 2) for pixel in values) / len(values)


def _uicm(image):
    red, green, blue = image[:, :, 0].flatten(), image[:, :, 1].flatten(), image[:, :, 2].flatten()
    red_green = red - green
    yellow_blue = ((red + green) / 2) - blue
    mean_rg, mean_yb = _trimmed_mean(red_green), _trimmed_mean(yellow_blue)
    var_rg, var_yb = _variance(red_green, mean_rg), _variance(yellow_blue, mean_yb)
    return (-0.0268 * math.sqrt(mean_rg**2 + mean_yb**2)) + (0.1586 * math.sqrt(var_rg + var_yb))


def _sobel(channel):
    dx, dy = ndimage.sobel(channel, 0), ndimage.sobel(channel, 1)
    magnitude = np.hypot(dx, dy)
    if np.max(magnitude) != 0:
        magnitude *= 255.0 / np.max(magnitude)
    return magnitude


def _eme(channel, window_size):
    horizontal, vertical = int(channel.shape[1] / window_size), int(channel.shape[0] / window_size)
    if horizontal * vertical == 0:
        return 0.0
    weight = 2.0 / (horizontal * vertical)
    channel = channel[: window_size * vertical, : window_size * horizontal]
    value = 0.0
    for column in range(horizontal):
        for row in range(vertical):
            block = channel[
                row * window_size : window_size * (row + 1),
                column * window_size : window_size * (column + 1),
            ]
            maximum, minimum = np.max(block), np.min(block)
            if minimum != 0.0 and maximum != 0.0:
                value += math.log(maximum / minimum)
    return weight * value


def _uism(image):
    red, green, blue = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    return (
        0.299 * _eme(np.multiply(_sobel(red), red), 10)
        + 0.587 * _eme(np.multiply(_sobel(green), green), 10)
        + 0.114 * _eme(np.multiply(_sobel(blue), blue), 10)
    )


def _uiconm(image, window_size):
    horizontal, vertical = int(image.shape[1] / window_size), int(image.shape[0] / window_size)
    if horizontal * vertical == 0:
        return 0.0
    weight = -1.0 / (horizontal * vertical)
    image = image[: window_size * vertical, : window_size * horizontal]
    value = 0.0
    for column in range(horizontal):
        for row in range(vertical):
            block = image[
                row * window_size : window_size * (row + 1),
                column * window_size : window_size * (column + 1),
                :,
            ]
            maximum, minimum = np.max(block), np.min(block)
            numerator, denominator = maximum - minimum, maximum + minimum
            if not (
                math.isnan(numerator)
                or math.isnan(denominator)
                or denominator == 0.0
                or numerator == 0.0
            ):
                value += (numerator / denominator) * math.log(numerator / denominator)
    return weight * value


def uiqm(image):
    image = image.astype(np.float32)
    return float(0.0282 * _uicm(image) + 0.2953 * _uism(image) + 3.5753 * _uiconm(image, 10))


def create_pyiqa_metrics(device):
    import pyiqa

    metrics = OrderedDict()
    metrics["psnr_y"] = pyiqa.create_metric("psnr", test_y_channel=True).to(device)
    metrics["psnr_rgb"] = pyiqa.create_metric("psnr").to(device)
    metrics["ssim_y"] = pyiqa.create_metric("ssim", test_y_channel=True).to(device)
    metrics["ssim_rgb"] = pyiqa.create_metric("ssim", test_y_channel=False).to(device)
    metrics["ms_ssim"] = pyiqa.create_metric("ms_ssim").to(device)
    metrics["lpips"] = pyiqa.create_metric("lpips").to(device)
    metrics["niqe"] = pyiqa.create_metric("niqe").to(device)
    metrics["musiq"] = pyiqa.create_metric("musiq").to(device)
    metrics["clipiqa"] = pyiqa.create_metric("clipiqa").to(device)
    return metrics


def _image_to_tensor_and_array(path: Path, eval_size: int, device):
    with Image.open(path) as source:
        image = source.convert("RGB")
    if eval_size:
        image = image.resize((eval_size, eval_size), Image.Resampling.LANCZOS)
    array = np.asarray(image)
    tensor = torch.from_numpy(array.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    return tensor, array


def _write_markdown(means: dict[str, float], path: Path, metric_names):
    lines = ["| Metric | Better | Mean |", "|---|---:|---:|"]
    for name in metric_names:
        direction = METRIC_DIRECTIONS[name]
        arrow = "↑" if direction == "higher" else "↓"
        lines.append(f"| {name.upper()} | {arrow} | {means[name]:.4f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_extended_metrics(
    prediction_dir: str | Path,
    reference_dir: str | Path,
    names: list[str],
    output_csv: str | Path,
    device: torch.device,
    eval_size: int = 256,
    cold_scores: dict[str, dict[str, float]] | None = None,
    pyiqa_metrics=None,
    progress_label: str = "extended_metrics",
):
    prediction_dir = Path(prediction_dir)
    reference_dir = Path(reference_dir)
    output_csv = Path(output_csv)
    if pyiqa_metrics is None:
        pyiqa_metrics = create_pyiqa_metrics(device)
    ducd = DUCD().to(device)
    scores = {name: [] for name in ALL_METRICS}
    rows = []

    for index, name in enumerate(names, start=1):
        prediction, prediction_array = _image_to_tensor_and_array(
            prediction_dir / f"{name}.png", eval_size, device
        )
        reference, reference_array = _image_to_tensor_and_array(
            reference_dir / f"{name}.png", eval_size, device
        )
        row = {"image": name}
        with torch.no_grad():
            for metric_name in ("niqe", "musiq", "clipiqa"):
                row[metric_name] = float(pyiqa_metrics[metric_name](prediction).item())
            row["ducd"] = float(ducd(prediction).item())
            for metric_name in ("psnr_y", "psnr_rgb", "ssim_y", "ssim_rgb", "ms_ssim", "lpips"):
                row[metric_name] = float(pyiqa_metrics[metric_name](prediction, reference).item())
        row["entropy"] = entropy(prediction_array)
        row["uiqm"] = uiqm(prediction_array)
        row["mi"] = mutual_information(reference_array, prediction_array)
        row["kl"] = kl_divergence(reference_array, prediction_array)
        if cold_scores is not None:
            row.update(cold_scores[name])
        for metric_name in ALL_METRICS:
            scores[metric_name].append(row[metric_name])
        rows.append(row)
        if index % 10 == 0 or index == len(names):
            print(f"{progress_label}={index}/{len(names)}")

    means = {name: float(np.mean(values)) for name, values in scores.items()}
    output_metrics = ALL_METRICS
    if cold_scores is not None:
        extra_metrics = tuple(next(iter(cold_scores.values())))
        output_metrics = ALL_METRICS + extra_metrics
        for metric_name in extra_metrics:
            means[metric_name] = float(np.mean([row[metric_name] for row in rows]))
    mean_row = {"image": "__mean__", **means}
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", *output_metrics])
        writer.writeheader()
        writer.writerows([*rows, mean_row])
    _write_markdown(means, output_csv.with_suffix(".md"), output_metrics)
    return {
        "num_images": len(names),
        "eval_size": eval_size,
        "means": means,
        "directions": METRIC_DIRECTIONS,
    }


def save_method_comparison(
    direct_means: dict[str, float],
    algorithm2_means: dict[str, float],
    output_csv: str | Path,
):
    """Compare direct reconstruction and Algorithm 2 from the same Cold model."""
    output_csv = Path(output_csv)
    metric_names = ALL_METRICS + ("delta_e76", "trajectory_monotonic")
    rows = []
    for metric_name in metric_names:
        direct_value = direct_means.get(metric_name)
        algorithm2_value = algorithm2_means.get(metric_name)
        direction = METRIC_DIRECTIONS[metric_name]
        if direct_value is None:
            winner = "Algorithm 2 diagnostic only"
            delta = None
        else:
            delta = algorithm2_value - direct_value
            if math.isclose(direct_value, algorithm2_value, rel_tol=1e-9, abs_tol=1e-12):
                winner = "tie"
            elif direction == "higher":
                winner = "Algorithm 2" if algorithm2_value > direct_value else "Direct"
            else:
                winner = "Algorithm 2" if algorithm2_value < direct_value else "Direct"
        rows.append(
            {
                "metric": metric_name,
                "better": direction,
                "direct": direct_value,
                "algorithm2": algorithm2_value,
                "algorithm2_minus_direct": delta,
                "winner": winner,
            }
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "metric",
        "better",
        "direct",
        "algorithm2",
        "algorithm2_minus_direct",
        "winner",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "| Metric | Better | Direct | Algorithm 2 | Winner |",
        "|---|---:|---:|---:|---|",
    ]
    for row in rows:
        arrow = "↑" if row["better"] == "higher" else "↓"
        direct_text = "N/A" if row["direct"] is None else f"{row['direct']:.4f}"
        algorithm2_text = f"{row['algorithm2']:.4f}"
        lines.append(
            f"| {row['metric'].upper()} | {arrow} | {direct_text} | "
            f"{algorithm2_text} | {row['winner']} |"
        )
    output_csv.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows
