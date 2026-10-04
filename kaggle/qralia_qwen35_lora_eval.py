"""Inférence Kaggle du LoRA QraLia sur le test SFT v0.4 gelé.

Inputs Kaggle requis :
- le Dataset privé ``qralia_sft_v0.4`` contenant ``test_prompts.jsonl`` ;
- l'output d'entraînement contenant ``qralia-qwen35-lora.zip`` (ou le
  dossier d'adaptateur décompressé).

Le script ne lance aucun entraînement et n'utilise jamais les cas de test
pour modifier le modèle. Il produit uniquement les prédictions brutes.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTHONUNBUFFERED"] = "1"

MODEL_ID = "Qwen/Qwen3.5-4B"
EXPECTED_TEST_COUNT = 160
MAX_NEW_TOKENS = 512
WORK_DIR = Path("/kaggle/working/qralia-lora-evaluation")
ADAPTER_DIR = WORK_DIR / "adapter"
OUTPUT_PATH = WORK_DIR / "qralia_lora_test_predictions.jsonl"
SUMMARY_PATH = WORK_DIR / "qralia_lora_test_run.json"


def run(command: list[str]) -> None:
    print("$", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unique_path(candidates: list[Path], label: str) -> Path:
    candidates = sorted(set(path.resolve() for path in candidates))
    if len(candidates) != 1:
        raise RuntimeError(f"{label}: un seul fichier attendu, trouvés={candidates}")
    return candidates[0]


def locate_test_prompts() -> Path:
    candidates = []
    for path in Path("/kaggle/input").glob("**/test_prompts.jsonl"):
        manifest_path = path.parent / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") == "0.4.0" and manifest.get("counts", {}).get("test") == EXPECTED_TEST_COUNT:
            candidates.append(path)
    return unique_path(candidates, "test_prompts.jsonl SFT v0.4")


def locate_adapter() -> tuple[Path, str]:
    zip_candidates = list(Path("/kaggle/input").glob("**/qralia-qwen35-lora.zip"))
    if zip_candidates:
        zip_path = unique_path(zip_candidates, "archive LoRA")
        if ADAPTER_DIR.exists():
            shutil.rmtree(ADAPTER_DIR)
        ADAPTER_DIR.mkdir(parents=True)
        shutil.unpack_archive(str(zip_path), str(ADAPTER_DIR))
        return ADAPTER_DIR, sha256(zip_path)

    config_candidates = [
        path
        for path in Path("/kaggle/input").glob("**/adapter_config.json")
        if (path.parent / "adapter_model.safetensors").exists()
    ]
    config_path = unique_path(config_candidates, "dossier LoRA")
    return config_path.parent, sha256(config_path.parent / "adapter_model.safetensors")


# Dépendances d'inférence seulement. Les kernels FLA restent exclus sur T4.
run([
    sys.executable,
    "-m",
    "pip",
    "install",
    "-qU",
    "transformers>=5.9,<6",
    "peft==0.19.1",
    "accelerate>=1.10,<2",
    "bitsandbytes>=0.49,<1",
    "qwen-vl-utils>=0.0.14",
    "decord",
])
run([
    sys.executable,
    "-m",
    "pip",
    "uninstall",
    "-y",
    "flash-linear-attention",
    "fla-core",
    "causal-conv1d",
])

import torch
from peft import PeftModel
from transformers import AutoModelForMultimodalLM, AutoProcessor, BitsAndBytesConfig


if not torch.cuda.is_available():
    raise RuntimeError("Activez un GPU Kaggle avant de lancer l'inférence.")

WORK_DIR.mkdir(parents=True, exist_ok=True)
test_prompts_path = locate_test_prompts()
adapter_dir, adapter_sha256 = locate_adapter()
test_rows = read_jsonl(test_prompts_path)
if len(test_rows) != EXPECTED_TEST_COUNT:
    raise RuntimeError(f"{EXPECTED_TEST_COUNT} tests attendus, trouvés={len(test_rows)}")
if len({row["case_id"] for row in test_rows}) != len(test_rows):
    raise RuntimeError("Les case_id du test doivent être uniques.")

print({
    "test_prompts": str(test_prompts_path),
    "adapter": str(adapter_dir),
    "adapter_sha256": adapter_sha256,
    "test_count": len(test_rows),
}, flush=True)

processor = AutoProcessor.from_pretrained(MODEL_ID)
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)
base_model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_ID,
    quantization_config=quantization_config,
    torch_dtype=torch.float16,
    device_map={"": 0},
    attn_implementation="eager",
    low_cpu_mem_usage=True,
)
model = PeftModel.from_pretrained(base_model, adapter_dir)
model.eval()

# Reprise possible si une exécution interactive a été interrompue.
existing = read_jsonl(OUTPUT_PATH) if OUTPUT_PATH.exists() else []
predictions = {row["case_id"]: row for row in existing}

for position, row in enumerate(test_rows, start=1):
    case_id = row["case_id"]
    if case_id in predictions:
        continue

    raw_messages = row["messages"]
    messages = [
        {
            "role": message["role"],
            "content": [{"type": "text", "text": message["content"]}],
        }
        for message in raw_messages
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    ).to("cuda:0")

    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
    prompt_tokens = inputs["input_ids"].shape[-1]
    completion_ids = generated[0, prompt_tokens:]
    raw_output = processor.decode(completion_ids, skip_special_tokens=True).strip()
    elapsed = time.perf_counter() - started

    try:
        json.loads(raw_output)
        valid_json = True
    except json.JSONDecodeError:
        valid_json = False

    predictions[case_id] = {
        "case_id": case_id,
        "raw_output": raw_output,
        "valid_json": valid_json,
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_ids.shape[-1]),
        "latency_seconds": elapsed,
    }
    ordered = [predictions[item["case_id"]] for item in test_rows if item["case_id"] in predictions]
    OUTPUT_PATH.write_text(
        "".join(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n" for item in ordered),
        encoding="utf-8",
    )
    print(
        f"[{position}/{len(test_rows)}] {case_id} "
        f"json={'OK' if valid_json else 'FAIL'} {elapsed:.1f}s",
        flush=True,
    )

summary = {
    "base_model": MODEL_ID,
    "adapter_sha256": adapter_sha256,
    "test_prompts_sha256": sha256(test_prompts_path),
    "case_count": len(predictions),
    "valid_json_count": sum(bool(item["valid_json"]) for item in predictions.values()),
    "max_new_tokens": MAX_NEW_TOKENS,
    "temperature": 0,
    "thinking": False,
    "test_used_for_training_or_tuning": False,
    "predictions": str(OUTPUT_PATH),
}
SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"Prédictions prêtes : {OUTPUT_PATH}")
print(f"Résumé : {SUMMARY_PATH}")
