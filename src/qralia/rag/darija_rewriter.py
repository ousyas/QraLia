"""Reformulation darija optionnelle avec verrouillage des valeurs sensibles."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Callable, Sequence
from urllib.error import URLError
from urllib.request import Request, urlopen


DEFAULT_ATLAS_MODEL = "atlasia/Al-Atlas-0.5B"
DEFAULT_ATLAS_CHAT_OLLAMA_MODEL = (
    "hf.co/tensorblock/MBZUAI-Paris_Atlas-Chat-2B-GGUF:latest"
)
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
DEFAULT_ATLAS_CACHE = Path("artifacts/models/huggingface")
REWRITE_PROMPT_PROFILES = {"baseline", "grounded_few_shot", "grounded_compact"}
NUMBER_PATTERN = re.compile(r"(?<!\w)\d+(?:[.,]\d+)?(?!\w)")
ARABIC_LETTER_PATTERN = re.compile(r"[\u0600-\u06ff]")
WORD_PATTERN = re.compile(r"\w+", re.UNICODE)


def values_to_protect(text: str, critical_values: Sequence[str]) -> list[str]:
    """Retourne les littéraux sensibles présents, du plus long au plus court."""
    candidates = [value for value in critical_values if value and value in text]
    candidates.extend(NUMBER_PATTERN.findall(text))
    return sorted(set(candidates), key=lambda value: (-len(value), value))


def lock_values(text: str, values: Sequence[str]) -> tuple[str, dict[str, str]]:
    """Remplace les valeurs par des marqueurs opaques avant reformulation."""
    locked = text
    mapping: dict[str, str] = {}
    for index, value in enumerate(values):
        if value not in locked:
            continue
        marker = f"ZXQVAL{index}QXZ"
        locked = locked.replace(value, marker)
        mapping[marker] = value
    return locked, mapping


def restore_values(text: str, mapping: dict[str, str]) -> str | None:
    """Restaure les valeurs, ou refuse une sortie qui perd un marqueur."""
    restored = text.strip()
    for marker, value in mapping.items():
        if restored.count(marker) != 1:
            return None
        restored = restored.replace(marker, value)
    return restored


def candidate_quality_issues(
    original: str,
    candidate: str,
    target_language: str = "darija_arab",
) -> list[str]:
    """Détecte seulement les corruptions évidentes, pas l'équivalence sémantique."""
    issues: list[str] = []
    if not candidate.strip():
        return ["empty_candidate"]
    if len(candidate) > max(240, 3 * len(original)):
        issues.append("candidate_too_long")
    words = [word.casefold() for word in WORD_PATTERN.findall(candidate)]
    if words and Counter(words).most_common(1)[0][1] > 4:
        issues.append("excessive_repetition")
    alphabetic = [character for character in candidate if character.isalpha()]
    if alphabetic:
        arabic_ratio = sum(bool(ARABIC_LETTER_PATTERN.fullmatch(character)) for character in alphabetic) / len(alphabetic)
        if target_language == "darija_arab" and arabic_ratio < 0.5:
            issues.append("insufficient_arabic_script")
        if target_language == "darija_latn" and arabic_ratio >= 0.5:
            issues.append("excessive_arabic_script")
    return issues


def build_rewrite_prompt(
    text: str,
    *,
    target_language: str,
    prompt_profile: str,
) -> str:
    """Construit un prompt de style ; le texte fourni reste l'unique fait."""
    if prompt_profile not in REWRITE_PROMPT_PROFILES:
        raise ValueError(f"Profil de reformulation inconnu : {prompt_profile}")
    if target_language not in {"darija_arab", "darija_latn"}:
        raise ValueError(f"Langue de reformulation non prise en charge : {target_language}")

    if target_language == "darija_arab":
        target_instruction = "بدارجة مغربية طبيعية وبالحروف العربية"
        examples = """أمثلة ديال الطريقة فقط، ماشي معلومات للجواب:
- الأصل: Vous avez ZXQVAL0QXZ jours pour payer.
  الصياغة: عندك ZXQVAL0QXZ أيام باش تخلّص.
- الأصل: La demande doit être remise à l'agence bancaire.
  الصياغة: خاص الطلب يتعطى للوكالة البنكية."""
    else:
        target_instruction = "بالدارجة المغربية وبالحروف اللاتينية"
        examples = """Exemples de style seulement, jamais des faits pour la réponse :
- Original : Vous avez ZXQVAL0QXZ jours pour payer.
  Reformulation : 3ndek ZXQVAL0QXZ iyam bach tkhelles.
- Original : La demande doit être remise à l'agence bancaire.
  Reformulation : Khass talab yt3ta l wakala bancaire."""

    if prompt_profile == "baseline":
        return (
            f"عاود كتب غير الجملة اللي تحت {target_instruction}. "
            "حافظ على نفس المعنى بلا ما تزيد ولا تنقص حتى معلومة. "
            "خلي الرموز اللي كيبداو بـ ZXQVAL كيف ما هما. "
            "جاوب غير بالجملة المعاود صياغتها، بلا شرح.\n\n"
            f"الجملة: {text}"
        )

    if prompt_profile == "grounded_compact":
        if target_language == "darija_latn":
            return f"""Reformule seulement le TEXTE en darija marocaine avec lettres latines.
Ne change, n'ajoute et ne supprime aucune information.
Copie exactement chaque nom, nombre et code ZXQVAL... une seule fois.
Réponds par la reformulation seule, sans titre ni explication.
Exemple: Vous avez ZXQVAL0QXZ jours. -> 3ndek ZXQVAL0QXZ iyam.
TEXTE: {text}"""
        return f"""عاود صيغ غير النص بالدارجة المغربية وبالحروف العربية.
ما تبدل، ما تزيد وما تحيد حتى معلومة.
انسخ كل اسم ورقم ورمز ZXQVAL... كيف ما هو ومرة وحدة.
جاوب غير بالصياغة، بلا عنوان وبلا شرح.
مثال: Vous avez ZXQVAL0QXZ jours. -> عندك ZXQVAL0QXZ أيام.
النص: {text}"""

    return f"""المهمة الوحيدة ديالك هي تعاود تصيغ النص {target_instruction}.

قواعد ضرورية:
- النص الأصلي هو المصدر الوحيد للمعلومات.
- حافظ على جميع المعاني، النفي، الشروط، والأفعال بلا زيادة ولا نقصان.
- ما تخترع حتى سبب، نتيجة، اسم، رقم ولا إجراء.
- خلي أسماء الشركات والخدمات والمختصرات التقنية كيف ما جاو إلا ما كانش عندها تعبير دارج واضح.
- كل رمز بحال ZXQVAL0QXZ قيمة محمية: خاصو يبقى بنفس الحروف، مرة وحدة، وفنفس المعنى.
- ما تكتبش جملة كاملة بالفرنسية؛ غير الأسماء التقنية يقدرو يبقاو باللاتينية.
- خرج غير الصياغة النهائية، بلا مقدمة وبلا شرح.

{examples}

النص اللي خاصك تعاود تصيغ دابا:
{text}"""


class AtlasDarijaRewriter:
    """Petit reformulateur local ; le backend est chargé paresseusement sur CPU."""

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_ATLAS_MODEL,
        cache_dir: Path = DEFAULT_ATLAS_CACHE,
        prompt_profile: str = "baseline",
        observation_mode: bool = False,
        backend: Callable[[str], str] | None = None,
    ) -> None:
        if prompt_profile not in REWRITE_PROMPT_PROFILES:
            raise ValueError(f"Profil de reformulation inconnu : {prompt_profile}")
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.prompt_profile = prompt_profile
        self.observation_mode = observation_mode
        self._backend = backend

    def _load_backend(self) -> Callable[[str], str]:
        if self._backend is not None:
            return self._backend

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, cache_dir=self.cache_dir)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            cache_dir=self.cache_dir,
            dtype=torch.float32,
        ).to("cpu")
        model.eval()

        def generate(prompt: str) -> str:
            messages = [{"role": "user", "content": prompt}]
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                    repetition_penalty=1.1,
                    pad_token_id=tokenizer.eos_token_id,
                )
            generated = outputs[0][inputs["input_ids"].shape[-1] :]
            return tokenizer.decode(generated, skip_special_tokens=True).strip()

        self._backend = generate
        return generate

    def rewrite(
        self,
        text: str,
        critical_values: Sequence[str],
        target_language: str = "darija_arab",
    ) -> tuple[str, dict[str, object]]:
        protected = values_to_protect(text, critical_values)
        locked, mapping = lock_values(text, protected)
        prompt = build_rewrite_prompt(
            locked,
            target_language=target_language,
            prompt_profile=self.prompt_profile,
        )
        candidate = self._load_backend()(prompt).strip().strip('"')
        restored = restore_values(candidate, mapping)
        observable_candidate = candidate
        for marker, value in mapping.items():
            observable_candidate = observable_candidate.replace(marker, value)
        if not restored:
            metadata = {
                "changed": observable_candidate != text,
                "fallback": not self.observation_mode,
                "would_fallback": True,
                "unsafe_observation": self.observation_mode,
                "reason": "protected_value_missing_or_duplicated",
                "protected_values": protected,
                "raw_candidate": candidate,
            }
            # Une chaîne vide ne peut pas former un AnswerClaim valide. On la
            # conserve dans raw_candidate, mais on garde le texte original.
            if self.observation_mode and observable_candidate.strip():
                return observable_candidate, metadata
            return text, {
                **metadata,
                "changed": False,
                "observation_applied": False,
            }
        quality_issues = candidate_quality_issues(text, restored, target_language)
        if quality_issues:
            metadata = {
                "changed": restored != text,
                "fallback": not self.observation_mode,
                "would_fallback": True,
                "unsafe_observation": self.observation_mode,
                "reason": "candidate_quality_gate_failed",
                "quality_issues": quality_issues,
                "protected_values": protected,
                "raw_candidate": candidate,
            }
            if self.observation_mode:
                return restored, metadata
            return text, {
                **metadata,
                "changed": False,
            }
        return restored, {
            "changed": restored != text,
            "fallback": False,
            "would_fallback": False,
            "unsafe_observation": self.observation_mode,
            "reason": None,
            "quality_issues": [],
            "protected_values": protected,
            "raw_candidate": candidate,
        }


class OllamaAtlasDarijaRewriter(AtlasDarijaRewriter):
    """Atlas-Chat quantifié via Ollama, avec les mêmes garde-fous de sortie."""

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_ATLAS_CHAT_OLLAMA_MODEL,
        url: str = DEFAULT_OLLAMA_URL,
        timeout_seconds: int = 180,
        prompt_profile: str = "baseline",
        observation_mode: bool = False,
        opener=urlopen,
    ) -> None:
        super().__init__(
            model_id=model_id,
            prompt_profile=prompt_profile,
            observation_mode=observation_mode,
        )
        self.url = url
        self.timeout_seconds = timeout_seconds
        self._opener = opener

    def _load_backend(self) -> Callable[[str], str]:
        if self._backend is not None:
            return self._backend

        def generate(prompt: str) -> str:
            payload = {
                "model": self.model_id,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "think": False,
                "options": {
                    "temperature": 0,
                    "seed": 42,
                    "num_ctx": 2048,
                    "num_predict": 128,
                },
                "keep_alive": "5m",
            }
            request = Request(
                self.url,
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with self._opener(request, timeout=self.timeout_seconds) as response:
                    raw = json.loads(response.read().decode("utf-8"))
            except URLError as exc:
                raise RuntimeError(
                    "Ollama est inaccessible pour la reformulation Atlas-Chat."
                ) from exc
            return str(raw["message"]["content"]).strip()

        self._backend = generate
        return generate
