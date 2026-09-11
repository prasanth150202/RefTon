"""RunPod serverless handler for RefTon (FLUX-Kontext virtual try-on).

Expected job["input"] fields:
  cloth        (str, required)  base64-encoded garment image
  mode         (str, optional)  "agnostic" (default) or "person"
  agnostic     (str, required if mode=="agnostic") base64 agnostic image
  person       (str, required if mode=="person")   base64 person image
  image_ref    (str, optional)  base64 reference-person image; enables the reference branch
  prompt       (str, optional)  default ""
  height       (int, optional)  default 512
  width        (int, optional)  default 384
  cond_scale   (float, optional) default 1.0 (use 2.0 for 1024x768)
  guidance_scale (float, optional) default 2.5
  num_inference_steps (int, optional) default 28
  seed         (int, optional)  default 42

Returns {"image": "<base64 png>"} on success, or {"error": "..."}.
"""

import base64
import io
import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

import runpod
from huggingface_hub import snapshot_download, hf_hub_download

from refton.pipelines import FluxKontextPipelineI2I

MODEL_DIR = os.environ.get("MODEL_DIR", "/runpod-volume/models/FLUX.1-Kontext-dev")
LORA_REPO = os.environ.get("LORA_REPO", "qihoo360/RefVTON")
LORA_DIR = os.environ.get("LORA_DIR", "/runpod-volume/models/RefVTON")
LORA_FILENAME = os.environ.get("LORA_FILENAME", "512_384_pytorch_lora_weights.safetensors")
HF_TOKEN = os.environ.get("HF_TOKEN")


def _ensure_weights():
    if not os.path.isdir(MODEL_DIR) or not os.listdir(MODEL_DIR):
        print(f"Downloading FLUX.1-Kontext-dev to {MODEL_DIR} ...")
        snapshot_download(
            "black-forest-labs/FLUX.1-Kontext-dev",
            local_dir=MODEL_DIR,
            token=HF_TOKEN,
        )
    lora_path = os.path.join(LORA_DIR, LORA_FILENAME)
    if not os.path.exists(lora_path):
        print(f"Downloading {LORA_FILENAME} from {LORA_REPO} ...")
        os.makedirs(LORA_DIR, exist_ok=True)
        hf_hub_download(
            LORA_REPO,
            filename=LORA_FILENAME,
            local_dir=LORA_DIR,
            token=HF_TOKEN,
        )
    return lora_path


print("Loading RefTon pipeline (cold start)...")
_lora_path = _ensure_weights()
pipe = FluxKontextPipelineI2I.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16)
pipe.load_lora_weights(_lora_path)
# 24GB-class GPUs can't hold FLUX-Kontext (transformer + T5-XXL) resident at once;
# this pages weights CPU<->GPU per step. Swap for pipe.to("cuda") on 48GB+ GPUs for speed.
pipe.enable_model_cpu_offload()
print("Pipeline ready.")


def _decode_image(b64_str):
    if "," in b64_str and b64_str.strip().startswith("data:"):
        b64_str = b64_str.split(",", 1)[1]
    raw = base64.b64decode(b64_str)
    img = Image.open(io.BytesIO(raw))
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def _build_transforms(size, cond_size):
    main_t = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    ref_t = transforms.Compose([
        transforms.Resize(cond_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(cond_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    return main_t, ref_t


def _tensor_to_png_b64(tensor):
    tensor = tensor.detach().cpu()
    if tensor.min() < 0:
        tensor = (tensor + 1) / 2
    array = tensor.float().numpy()
    array = np.transpose(array, (1, 2, 0))
    array = (array * 255).clip(0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(array).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def handler(job):
    inp = job.get("input", {})

    try:
        if "cloth" not in inp:
            return {"error": "missing required field: cloth"}

        mode = inp.get("mode", "agnostic")
        use_person = mode == "person"
        use_reference = "image_ref" in inp and inp["image_ref"]

        height = int(inp.get("height", 512))
        width = int(inp.get("width", 384))
        cond_scale = float(inp.get("cond_scale", 1.0))
        size = (height, width)
        cond_size = [int(x // cond_scale) for x in size]
        main_t, ref_t = _build_transforms(size, cond_size)

        torch.manual_seed(int(inp.get("seed", 42)))

        batch = {"cond_pixel_values_cloth": main_t(_decode_image(inp["cloth"])).unsqueeze(0)}

        if use_person:
            if "person" not in inp:
                return {"error": "mode=person requires field: person"}
            batch["cond_pixel_values_person"] = main_t(_decode_image(inp["person"])).unsqueeze(0)
            key_to_index_scale = {
                "cond_pixel_values_person": [1, 1],
                "cond_pixel_values_cloth": [2, 1],
            }
        else:
            if "agnostic" not in inp:
                return {"error": "mode=agnostic requires field: agnostic"}
            batch["cond_pixel_values_agnostic"] = main_t(_decode_image(inp["agnostic"])).unsqueeze(0)
            key_to_index_scale = {
                "cond_pixel_values_agnostic": [1, 1],
                "cond_pixel_values_cloth": [2, 1],
            }

        if use_reference:
            batch["pixel_values_ref"] = ref_t(_decode_image(inp["image_ref"])).unsqueeze(0)
            key_to_index_scale["pixel_values_ref"] = [5, cond_scale]

        with torch.no_grad():
            images = pipe(
                image=batch,
                batch_size=1,
                prompt=inp.get("prompt", ""),
                num_images_per_prompt=1,
                guidance_scale=float(inp.get("guidance_scale", 2.5)),
                num_inference_steps=int(inp.get("num_inference_steps", 28)),
                generator=torch.Generator().manual_seed(int(inp.get("seed", 42))),
                height=height,
                width=width,
                cond_scale=cond_scale,
                key_to_index_scale=key_to_index_scale,
            ).images

        return {"image": _tensor_to_png_b64(images[0])}

    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}


runpod.serverless.start({"handler": handler})
