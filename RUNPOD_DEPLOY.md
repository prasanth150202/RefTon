# Deploying RefTon on RunPod Serverless

Files added for this: `Dockerfile`, `runpod_handler.py`, `.dockerignore`.

## 1. Prerequisites

- A RunPod account + API key (Settings → API Keys).
- Access to the gated model on Hugging Face: request access at
  https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev, then create an
  HF access token (read scope) at https://huggingface.co/settings/tokens.
- A container registry to push to (Docker Hub, GHCR, etc.) — RunPod pulls the
  image from there.

## 2. Build and push the image

```bash
docker build -t <your-registry>/<your-user>/refton-runpod:latest .
docker push <your-registry>/<your-user>/refton-runpod:latest
```

The image itself does **not** bundle the FLUX-Kontext weights (~24GB) or the
RefTon LoRA — they're downloaded on first cold start into a RunPod Network
Volume and reused after that. This keeps the image small and avoids
re-uploading tens of GB every time you rebuild.

## 3. Create a Network Volume

In the RunPod console: Storage → New Network Volume. Pick a region that has
your target GPU available. 50GB is enough for the FLUX-Kontext weights + LoRA
+ HF cache.

## 4. Create the Serverless endpoint

Serverless → New Endpoint:

- Container image: the one you pushed above.
- GPU: 24GB tier (L4 / RTX 3090 / RTX 4090) per your earlier choice.
- Attach the Network Volume from step 3, mounted at `/runpod-volume`
  (RunPod's default mount path — the handler already reads/writes model
  weights there).
- Environment variables:
  - `HF_TOKEN` = your Hugging Face token (needed to download the gated FLUX
    weights on first cold start).
  - Optional overrides (defaults shown): `MODEL_DIR=/runpod-volume/models/FLUX.1-Kontext-dev`,
    `LORA_REPO=qihoo360/RefVTON`, `LORA_DIR=/runpod-volume/models/RefVTON`,
    `LORA_FILENAME=512_384_pytorch_lora_weights.safetensors`.
- Container disk: 20GB+ (for the image itself, layers, pip cache).

**Important — verify the LoRA filename before relying on defaults.** The
handler defaults to `512_384_pytorch_lora_weights.safetensors`, inferred from
the example commands in this repo's README, but the actual filename(s) in
https://huggingface.co/qihoo360/RefVTON may differ (e.g. a `1024_768_...`
variant, or a subfolder). Check the repo's file list and set `LORA_FILENAME`
(and `LORA_REPO`/`LORA_DIR` if needed) accordingly.

The first request after deploying will be slow (potentially 15-30+ minutes)
because it downloads and caches the model weights to the network volume.
Subsequent cold starts reuse the cached files and are much faster.

## 5. 24GB GPU caveat

FLUX.1-Kontext-dev's transformer (~12B params) plus its T5-XXL text encoder
don't fit resident in 24GB of VRAM at once in bf16. `runpod_handler.py` calls
`pipe.enable_model_cpu_offload()` to page weights between CPU RAM and GPU per
step, which is what makes 24GB workable — at the cost of extra latency per
request. If you move to a 48GB+ GPU later, swap that line for `pipe.to("cuda")`
for meaningfully faster inference.

## 6. Test the endpoint

```bash
python - <<'EOF'
import base64, requests, json

def b64(path):
    return base64.b64encode(open(path, "rb").read()).decode()

payload = {
    "input": {
        "mode": "agnostic",
        "cloth": b64("example/cloth/000001_1.jpg"),
        "agnostic": b64("example/agnostic/000001_0.jpg"),
        "height": 512,
        "width": 384,
    }
}

r = requests.post(
    "https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync",
    headers={"Authorization": "Bearer <RUNPOD_API_KEY>"},
    json=payload,
    timeout=600,
)
resp = r.json()
if "output" in resp and "image" in resp["output"]:
    open("result.png", "wb").write(base64.b64decode(resp["output"]["image"]))
    print("saved result.png")
else:
    print(json.dumps(resp, indent=2))
EOF
```

Swap in real file paths from `example/` (or your own images) and the actual
`<ENDPOINT_ID>` / `<RUNPOD_API_KEY>`.

## Request/response reference

See the docstring at the top of `runpod_handler.py` for the full list of
`input` fields (`cloth`, `mode`, `agnostic`/`person`, `image_ref`, `prompt`,
`height`, `width`, `cond_scale`, `guidance_scale`, `num_inference_steps`,
`seed`). Response is `{"image": "<base64 PNG>"}` or `{"error": "..."}`.
