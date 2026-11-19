"""Build a printable mesh of the reference figurine with Hunyuan3D-2.

The shape model runs on Apple Metal. Texture is unnecessary: the print is
one solid green part. Output is a raw mesh; make_printable.py fits it to
the bed and writes the STL / 3MF.

The upstream checkout lives in .tools/Hunyuan3D-2 (gitignored). After a
fresh clone of https://github.com/Tencent-Hunyuan/Hunyuan3D-2 into that
path, apply patches/hunyuan3d-apple.patch so sampling runs on MPS.
"""
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.85")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.5")

import sys
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / ".tools" / "Hunyuan3D-2"))

from hy3dgen.shapegen.pipelines import Hunyuan3DDiTFlowMatchingPipeline  # noqa: E402


def fetch_shape_weights():
    local = Path.home() / ".cache" / "hy3dgen" / "tencent" / "Hunyuan3D-2"
    local.mkdir(parents=True, exist_ok=True)
    for name in (
        "hunyuan3d-dit-v2-0/config.yaml",
        "hunyuan3d-dit-v2-0/model.fp16.safetensors",
    ):
        print("fetch", name, flush=True)
        hf_hub_download("tencent/Hunyuan3D-2", name, local_dir=str(local))


def main():
    fetch_shape_weights()
    print("loading pipeline", flush=True)
    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        "tencent/Hunyuan3D-2",
        subfolder="hunyuan3d-dit-v2-0",
        variant="fp16",
        use_safetensors=True,
        device="mps",
        dtype=torch.float16,
    )
    # Hierarchical decode. Keep the full VAE; the turbo decoder loses detail.
    pipe.enable_flashvdm(enabled=True, mc_algo="mc", replace_vae=False)

    generator = torch.Generator(device="cpu").manual_seed(42)
    print("sampling", flush=True)
    mesh = pipe(
        image=str(ROOT / "reference_cutout.png"),
        num_inference_steps=30,
        octree_resolution=384,
        guidance_scale=5.0,
        num_chunks=4000,
        generator=generator,
    )[0]
    if mesh is None:
        raise RuntimeError("surface extraction returned no mesh")

    print(
        "verts", len(mesh.vertices),
        "faces", len(mesh.faces),
        "watertight", mesh.is_watertight,
        "bounds", mesh.bounds.tolist(),
        flush=True,
    )
    mesh.export(ROOT / "army_man_raw.glb")
    mesh.export(ROOT / "army_man_raw.stl")
    print("saved army_man_raw.glb", flush=True)


if __name__ == "__main__":
    main()
