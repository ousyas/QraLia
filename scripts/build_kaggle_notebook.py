"""Génère un .ipynb Kaggle sans dupliquer manuellement le script source."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = (
    (
        ROOT / "kaggle" / "qralia_qwen35_qlora.py",
        ROOT / "kaggle" / "qralia_qwen35_qlora.ipynb",
        "# QraLia — QLoRA de Qwen3.5-4B\n",
        "Activez un GPU et ajoutez le Dataset privé relu `qralia_sft_v0.4` avant l'exécution.\n",
    ),
    (
        ROOT / "kaggle" / "qralia_qwen35_lora_eval.py",
        ROOT / "kaggle" / "qralia_qwen35_lora_eval.ipynb",
        "# QraLia — Évaluation du LoRA Qwen3.5-4B\n",
        "Activez un GPU et ajoutez `qralia_sft_v0.4` ainsi que l'output LoRA du notebook d'entraînement.\n",
    ),
)


def main() -> None:
    for source, output, title, instructions in NOTEBOOKS:
        code = source.read_text(encoding="utf-8")
        notebook = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        title,
                        "\n",
                        f"Notebook reproductible généré depuis `{source.relative_to(ROOT)}`. ",
                        instructions,
                    ],
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": code.splitlines(keepends=True),
                },
            ],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                },
                "language_info": {"name": "python", "version": "3"},
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        output.write_text(
            json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8",
        )
        print(output)


if __name__ == "__main__":
    main()
