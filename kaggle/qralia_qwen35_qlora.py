"""Notebook-script Kaggle pour le SFT QLoRA de Qwen3.5-4B.

Dans Kaggle :
1. activer un GPU ;
2. ajouter le dossier relu ``qralia_sft_v0.4`` comme Dataset privé ;
3. copier ce fichier dans une cellule ou l'exécuter comme script ;
4. enregistrer une version du notebook pour conserver /kaggle/working.

Le test gelé n'est volontairement jamais passé à ``swift sft``.
"""

# %% [markdown]
# # QraLia — QLoRA de Qwen3.5-4B
# Le notebook entraîne uniquement un adaptateur LoRA sur `train.jsonl`, choisit
# les réglages avec `validation.jsonl` et laisse `test_prompts.jsonl` fermé.

# %%
from __future__ import annotations

import hashlib
import json
import os

# Kaggle propose parfois deux T4, mais ce notebook utilise volontairement un
# seul processus et un seul GPU. Cela évite que MS-SWIFT tente une stratégie
# multi-GPU implicite alors que le modèle QLoRA tient sur une T4.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["IMAGE_MAX_TOKEN_NUM"] = "1024"
os.environ["VIDEO_MAX_TOKEN_NUM"] = "128"
os.environ["FPS_MAX_FRAMES"] = "12"
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["SWIFT_DEBUG"] = "1"

import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import distribution
from pathlib import Path


MODEL_ID = "Qwen/Qwen3.5-4B"
SEED = 42
MAX_LENGTH = 2048
EPOCHS = 2
LEARNING_RATE = "1e-4"
LORA_RANK = 8
LORA_ALPHA = 16
GRADIENT_ACCUMULATION_STEPS = 16

# Les kernels FLA/Triton accélèrent GatedDeltaNet sur des GPU récents, mais
# l'essai Kaggle a fait redémarrer le noyau pendant leur compilation sur T4
# (compute capability 7.5). Le chemin PyTorch eager est plus lent, mais stable
# et permet de vérifier l'entraînement avant toute optimisation matérielle.
USE_FLA_KERNELS = False

WORK_DIR = Path("/kaggle/working/qralia-qlora")
PREPARED_DIR = WORK_DIR / "prepared-data"
OUTPUT_DIR = WORK_DIR / "training-output"


def run(command: list[str]) -> None:
    print("$", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def tail_text(path: Path, max_bytes: int = 16_384) -> str:
    """Lit uniquement la fin d'un journal potentiellement volumineux."""
    with path.open("rb") as reader:
        reader.seek(0, 2)
        size = reader.tell()
        reader.seek(max(0, size - max_bytes))
        return reader.read().decode("utf-8", errors="replace")


def run_logged(command: list[str], log_path: Path) -> None:
    """Exécute l'entraînement sans saturer le canal de sortie de Kaggle."""
    print("$", " ".join(command), flush=True)
    with log_path.open("w", encoding="utf-8") as writer:
        process = subprocess.Popen(
            command,
            stdout=writer,
            stderr=subprocess.STDOUT,
            text=True,
        )
        while process.poll() is None:
            time.sleep(20)
            lines = [line for line in tail_text(log_path).replace("\r", "\n").splitlines() if line.strip()]
            last_line = lines[-1] if lines else "initialisation"
            print(f"[swift] {log_path.stat().st_size / 1024**2:.1f} MiB — {last_line[-300:]}", flush=True)

    if process.returncode != 0:
        print(tail_text(log_path, max_bytes=65_536), flush=True)
        raise subprocess.CalledProcessError(process.returncode, command)

    print(tail_text(log_path, max_bytes=65_536), flush=True)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def locate_dataset() -> Path:
    configured = os.environ.get("QRALIA_DATASET_DIR")
    if configured:
        candidate = Path(configured)
        if (candidate / "manifest.json").exists():
            return candidate
        raise FileNotFoundError(f"Manifest introuvable dans {candidate}")

    candidates = sorted(Path("/kaggle/input").glob("**/manifest.json"))
    valid = [
        path.parent
        for path in candidates
        if (path.parent / "train.jsonl").exists()
        and (path.parent / "validation.jsonl").exists()
        and (path.parent / "test_prompts.jsonl").exists()
    ]
    if len(valid) != 1:
        raise RuntimeError(
            "Ajoutez le Dataset Kaggle qralia_sft_v0.4 relu ou définissez "
            "QRALIA_DATASET_DIR. Candidats trouvés: " + repr(valid)
        )
    return valid[0]


def verify_manifest(dataset_dir: Path) -> dict:
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    for metadata in manifest["files"].values():
        path = dataset_dir / metadata["path"]
        actual = sha256(path)
        if actual != metadata["sha256"]:
            raise ValueError(f"SHA-256 invalide pour {path.name}: {actual}")
    if manifest["status"] != "human_reviewed":
        raise RuntimeError(
            "Le dataset est encore candidate_human_review. Faites valider les "
            "réponses puis changez explicitement son statut avec le script de revue."
        )
    return manifest


def prepare_swift_jsonl(source: Path, destination: Path) -> int:
    """Ne donne à MS-SWIFT que les messages, sans métadonnées d'audit."""
    count = 0
    with source.open(encoding="utf-8") as reader, destination.open("w", encoding="utf-8") as writer:
        for line in reader:
            record = json.loads(line)
            messages = record["messages"]
            if [message["role"] for message in messages] != ["user", "assistant"]:
                raise ValueError(f"Conversation invalide: {record['sample_id']}")
            json.loads(messages[1]["content"])
            writer.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
            count += 1
    return count


# %% [markdown]
# ## Installation
# Qwen3.5 requiert une version récente de Transformers. MS-SWIFT fournit le
# chargement Qwen3.5 et le QLoRA bitsandbytes.

# %%
run([
    sys.executable,
    "-m",
    "pip",
    "install",
    "-qU",
    "ms-swift>=4.3.1,<5",
    "transformers>=5.9,<6",
    "bitsandbytes>=0.46,<1",
    "qwen-vl-utils>=0.0.14",
    "decord",
])

# Qwen3.5 utilise des couches GatedDeltaNet. Les kernels optionnels FLA sont
# utiles sur un GPU compatible, mais restent désactivés par défaut sur T4.
if USE_FLA_KERNELS:
    run([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-U",
        "flash-linear-attention>=0.4.2",
        "causal-conv1d",
        "--no-build-isolation",
    ])
else:
    # Une session Kaggle réutilisée peut encore contenir les kernels d'un essai
    # précédent. Leur retrait force Transformers à choisir son implémentation
    # PyTorch au prochain processus d'entraînement.
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

# Résoudre explicitement le snapshot avant de lancer MS-SWIFT. Cela rend le
# téléchargement visible et évite qu'un accès au Hub reste bloqué sans journal.
from huggingface_hub import snapshot_download

MODEL_PATH = Path(snapshot_download(MODEL_ID))
print({"base_model": MODEL_ID, "local_snapshot": str(MODEL_PATH)}, flush=True)

SWIFT_SFT_SCRIPT = Path(distribution("ms-swift").locate_file("swift/cli/sft.py"))
if not SWIFT_SFT_SCRIPT.is_file():
    raise RuntimeError("Entrée Python swift/cli/sft.py introuvable après installation de ms-swift.")

run(["nvidia-smi"])

# %% [markdown]
# ## Vérification et préparation
# Le statut `human_reviewed` et tous les SHA-256 sont vérifiés avant qu'un seul
# pas d'entraînement puisse démarrer.

# %%
dataset_dir = locate_dataset()
manifest = verify_manifest(dataset_dir)
PREPARED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

train_path = PREPARED_DIR / "train_swift.jsonl"
validation_path = PREPARED_DIR / "validation_swift.jsonl"
train_count = prepare_swift_jsonl(dataset_dir / "train.jsonl", train_path)
validation_count = prepare_swift_jsonl(dataset_dir / "validation.jsonl", validation_path)

assert train_count == manifest["counts"]["train"]
assert validation_count == manifest["counts"]["validation"]
print({"dataset": str(dataset_dir), "train": train_count, "validation": validation_count})

# %% [markdown]
# ## Entraînement QLoRA
# Les T4 ne supportent pas nativement BF16 : calcul et LoRA utilisent FP16.
# La tokenisation est faite avant l'entraînement, sans workers persistants,
# pour éviter le blocage observé avec le template multimodal de Qwen3.5.

# %%
command = [
    "timeout", "--signal=INT", "11h",
    sys.executable, "-u", str(SWIFT_SFT_SCRIPT),
    "--model", str(MODEL_PATH),
    "--dataset", str(train_path),
    "--val_dataset", str(validation_path),
    "--split_dataset_ratio", "0",
    "--tuner_type", "lora",
    "--quant_method", "bnb",
    "--quant_bits", "4",
    "--bnb_4bit_quant_type", "nf4",
    "--bnb_4bit_use_double_quant", "true",
    "--torch_dtype", "float16",
    "--bnb_4bit_compute_dtype", "float16",
    "--attn_impl", "eager",
    "--freeze_vit", "true",
    "--freeze_aligner", "true",
    "--target_modules", "all-linear",
    "--lora_rank", str(LORA_RANK),
    "--lora_alpha", str(LORA_ALPHA),
    "--lora_dropout", "0",
    "--add_non_thinking_prefix", "true",
    "--loss_scale", "ignore_empty_think",
    "--num_train_epochs", str(EPOCHS),
    "--per_device_train_batch_size", "1",
    "--per_device_eval_batch_size", "1",
    "--gradient_accumulation_steps", str(GRADIENT_ACCUMULATION_STEPS),
    "--gradient_checkpointing", "true",
    "--learning_rate", LEARNING_RATE,
    "--warmup_ratio", "0.05",
    "--max_length", str(MAX_LENGTH),
    "--truncation_strategy", "delete",
    "--eval_steps", "10",
    "--save_steps", "10",
    "--logging_steps", "1",
    "--save_total_limit", "2",
    "--save_only_model", "true",
    "--dataset_num_proc", "1",
    "--lazy_tokenize", "false",
    "--dataloader_num_workers", "0",
    "--packing", "false",
    "--padding_free", "false",
    "--use_logits_to_keep", "true",
    "--dataloader_persistent_workers", "false",
    "--seed", str(SEED),
    "--data_seed", str(SEED),
    "--output_dir", str(OUTPUT_DIR),
    "--report_to", "none",
]

(WORK_DIR / "training_command.json").write_text(
    json.dumps(command, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
run_logged(command, WORK_DIR / "training.log")

# %% [markdown]
# ## Export de l'adaptateur
# On archive uniquement le checkpoint LoRA le plus récent et les informations
# nécessaires à la reproduction. Le test gelé reste inutilisé à ce stade.

# %%
checkpoints = sorted(
    OUTPUT_DIR.glob("**/checkpoint-*"),
    key=lambda path: path.stat().st_mtime,
)
if not checkpoints:
    raise RuntimeError(f"Aucun checkpoint trouvé dans {OUTPUT_DIR}")
latest_checkpoint = checkpoints[-1]

run_manifest = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "base_model": MODEL_ID,
    "dataset_manifest_sha256": sha256(dataset_dir / "manifest.json"),
    "dataset_counts": manifest["counts"],
    "test_used_during_training": False,
    "seed": SEED,
    "max_length": MAX_LENGTH,
    "epochs": EPOCHS,
    "learning_rate": LEARNING_RATE,
    "lora_rank": LORA_RANK,
    "lora_alpha": LORA_ALPHA,
    "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
    "use_fla_kernels": USE_FLA_KERNELS,
    "checkpoint": str(latest_checkpoint),
}
(WORK_DIR / "run_manifest.json").write_text(
    json.dumps(run_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)

export_dir = WORK_DIR / "qralia-qwen35-lora"
if export_dir.exists():
    shutil.rmtree(export_dir)
shutil.copytree(latest_checkpoint, export_dir)
shutil.copy2(WORK_DIR / "run_manifest.json", export_dir / "qralia_run_manifest.json")
shutil.copy2(dataset_dir / "manifest.json", export_dir / "qralia_dataset_manifest.json")
archive = shutil.make_archive(str(export_dir), "zip", root_dir=export_dir)
print(f"Adaptateur prêt: {archive}")
print("Dans Kaggle, utilisez Save Version puis téléchargez le ZIP depuis Output.")
