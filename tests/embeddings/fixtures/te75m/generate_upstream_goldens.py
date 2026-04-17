#!/usr/bin/env python3

from __future__ import annotations

import argparse
import contextlib
import io
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors.torch import load_file
from sentence_transformers import SentenceTransformer
import timm
from timm.data import create_transform, resolve_model_data_config


CPU = torch.device("cpu")
TE_REPO = "augmem/TE-75M"
TE_FILE = "TE-75M.safetensors"
EFFICIENTAT_GIT = "https://github.com/fschmid56/EfficientAT"


class ProjectionHead(nn.Module):
  def __init__(self, state: dict[str, torch.Tensor], prefix: str, encoder_dim: int):
    super().__init__()
    self.expand = nn.Linear(encoder_dim, 1920)
    self.expand_norm = nn.LayerNorm(1920)
    self.residual = nn.Linear(1920, 1920)
    self.residual_norm = nn.LayerNorm(1920)
    self.project = nn.Linear(1920, 1280)
    self.load_state_dict(
      {
        "expand.weight": state[f"{prefix}.expand.0.weight"],
        "expand.bias": state[f"{prefix}.expand.0.bias"],
        "expand_norm.weight": state[f"{prefix}.expand.2.weight"],
        "expand_norm.bias": state[f"{prefix}.expand.2.bias"],
        "residual.weight": state[f"{prefix}.residual_blocks.0.0.weight"],
        "residual.bias": state[f"{prefix}.residual_blocks.0.0.bias"],
        "residual_norm.weight": state[f"{prefix}.residual_blocks.0.2.weight"],
        "residual_norm.bias": state[f"{prefix}.residual_blocks.0.2.bias"],
        "project.weight": state[f"{prefix}.project.weight"],
        "project.bias": state[f"{prefix}.project.bias"],
      }
    )
    self.to(CPU)
    self.eval()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.expand(x)
    x = F.gelu(x)
    x = self.expand_norm(x)
    residual = x
    x = self.residual(x)
    x = F.gelu(x)
    x = self.residual_norm(x)
    x = x + residual
    x = self.project(x)
    return F.normalize(x, dim=-1)


def ensure_efficientat_clone(path: Path) -> Path:
  if path.exists():
    return path
  path.parent.mkdir(parents=True, exist_ok=True)
  subprocess.run(
    ["git", "clone", "--depth=1", EFFICIENTAT_GIT, str(path)],
    check=True,
  )
  return path


@contextlib.contextmanager
def pushd(path: Path):
  previous = Path.cwd()
  os.chdir(path)
  try:
    yield
  finally:
    os.chdir(previous)


def load_text_encoder(state: dict[str, torch.Tensor]) -> tuple[SentenceTransformer, ProjectionHead]:
  model = SentenceTransformer("MongoDB/mdbr-leaf-ir", device="cpu")
  text_sd = {}
  for key, value in state.items():
    if key.startswith("text_encoder."):
      text_sd[key.removeprefix("text_encoder.").replace("0.auto_model.", "0.model.")] = value
  missing, unexpected = model.load_state_dict(text_sd, strict=False)
  if missing or unexpected:
    raise RuntimeError(f"text encoder state mismatch: missing={missing} unexpected={unexpected}")
  model.eval()
  return model, ProjectionHead(state, "text_projection", 768)


def load_image_encoder(state: dict[str, torch.Tensor]) -> tuple[nn.Module, ProjectionHead, callable]:
  image_model = timm.create_model(
    "mobilenetv4_conv_medium.e180_r384_in12k",
    pretrained=False,
  ).to(CPU)
  image_sd = {
    key.removeprefix("image_encoder."): value for key, value in state.items()
    if key.startswith("image_encoder.")
  }
  missing, unexpected = image_model.load_state_dict(image_sd, strict=False)
  if missing != ["classifier.weight", "classifier.bias"] or unexpected:
    raise RuntimeError(f"image encoder state mismatch: missing={missing} unexpected={unexpected}")
  image_model.eval()
  image_transform = create_transform(
    **resolve_model_data_config(image_model),
    is_training=False,
  )
  return image_model, ProjectionHead(state, "image_projection", 1280), image_transform


def load_audio_encoder(
    state: dict[str, torch.Tensor],
    efficientat_root: Path,
) -> tuple[nn.Module, ProjectionHead, nn.Module]:
  sys.path.insert(0, str(efficientat_root))
  with pushd(efficientat_root):
    from models.preprocess import AugmentMelSTFT  # pylint: disable=import-error
    with contextlib.redirect_stdout(io.StringIO()):
      from models.mn.model import get_model as get_mn  # pylint: disable=import-error
      audio_model = get_mn(width_mult=2.0, pretrained_name=None, num_classes=527).to(CPU)
  audio_sd = {
    key.removeprefix("audio_encoder."): value for key, value in state.items()
    if key.startswith("audio_encoder.")
  }
  missing, unexpected = audio_model.load_state_dict(audio_sd, strict=False)
  if missing or unexpected:
    raise RuntimeError(f"audio encoder state mismatch: missing={missing} unexpected={unexpected}")
  audio_model.eval()
  audio_pre = AugmentMelSTFT(
    n_mels=128,
    sr=32000,
    win_length=800,
    hopsize=320,
    n_fft=1024,
    freqm=0,
    timem=0,
    fmin=0.0,
    fmax=15000,
    fmin_aug_range=1,
    fmax_aug_range=1,
  ).to(CPU)
  audio_pre.eval()
  return audio_model, ProjectionHead(state, "audio_projection", 1920), audio_pre


def synthesize_red_square() -> Image.Image:
  rgba = np.zeros((32, 32, 4), dtype=np.uint8)
  rgba[..., 0] = 255
  rgba[..., 3] = 255
  return Image.fromarray(rgba, mode="RGBA").convert("RGB")


def synthesize_pure_tone() -> torch.Tensor:
  time = torch.arange(4000, dtype=torch.float32, device=CPU)
  samples = 0.2 * torch.sin((2.0 * math.pi * 440.0 * time) / 16000.0)
  return torchaudio.functional.resample(samples.unsqueeze(0), 16000, 32000)


def load_state() -> dict[str, torch.Tensor]:
  te_path = hf_hub_download(repo_id=TE_REPO, filename=TE_FILE)
  return load_file(te_path, device="cpu")


def cosine(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
  return float(torch.dot(lhs, rhs))


def generate_goldens(output_dir: Path, efficientat_root: Path) -> None:
  state = load_state()
  text_model, text_projection = load_text_encoder(state)
  image_model, image_projection, image_transform = load_image_encoder(state)
  audio_model, audio_projection, audio_pre = load_audio_encoder(state, efficientat_root)

  red_square_text = (output_dir / "red-square.txt").read_text(encoding="utf-8")
  pure_tone_text = (output_dir / "pure-tone-440hz.txt").read_text(encoding="utf-8")
  image_tensor = image_transform(synthesize_red_square()).unsqueeze(0).to(CPU)
  audio_wave = synthesize_pure_tone()

  with torch.no_grad():
    text_embeddings = text_model.encode(
      [red_square_text, pure_tone_text],
      convert_to_tensor=True,
      device="cpu",
    ).to(CPU)
    text_red = text_projection(text_embeddings[0].unsqueeze(0)).squeeze(0)
    text_tone = text_projection(text_embeddings[1].unsqueeze(0)).squeeze(0)

    image_features = image_model.forward_features(image_tensor)
    image_preprojection = image_model.forward_head(image_features, pre_logits=True)
    image_red = image_projection(image_preprojection).squeeze(0)

    audio_mel = audio_pre(audio_wave).unsqueeze(1)
    _, audio_features = audio_model(audio_mel)
    audio_tone = audio_projection(audio_features).squeeze(0)

  vectors = {
    "red-square.text.1280.txt": text_red,
    "pure-tone-440hz.text.1280.txt": text_tone,
    "red-square.image.1280.txt": image_red,
    "pure-tone-440hz.audio.1280.txt": audio_tone,
  }
  for filename, vector in vectors.items():
    np.savetxt(output_dir / filename, vector.cpu().numpy(), fmt="%.9e")

  print("stored upstream TE goldens:")
  for filename, vector in vectors.items():
    print(f"  {filename}: norm={torch.linalg.vector_norm(vector).item():.6f}")
  print(f"  cosine(text_red, image_red)={cosine(text_red, image_red):.6f}")
  print(f"  cosine(text_red, audio_tone)={cosine(text_red, audio_tone):.6f}")
  print(f"  cosine(text_tone, audio_tone)={cosine(text_tone, audio_tone):.6f}")
  print(f"  cosine(text_tone, image_red)={cosine(text_tone, image_red):.6f}")


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--output-dir",
    type=Path,
    default=Path(__file__).resolve().parent,
  )
  parser.add_argument(
    "--efficientat-root",
    type=Path,
    default=Path(tempfile.gettempdir()) / "EfficientAT",
  )
  args = parser.parse_args()

  ensure_efficientat_clone(args.efficientat_root)
  generate_goldens(args.output_dir.resolve(), args.efficientat_root.resolve())
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
