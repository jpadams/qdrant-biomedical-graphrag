"""HuggingFace transformers-based biomedical NER extractor (maximal mode).

Uses token classification models from HuggingFace Hub. No scispaCy dependency.
Runs two complementary models by default:
- raynardj/ner-gene-dna-rna-jnlpba-pubmed: proteins, DNA, RNA, cell types
- d4data/biomedical-ner-all: diseases, medications, anatomy, procedures
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

from biomedical_graphrag.extraction.base import ExtractedEntity, ExtractionResult
from biomedical_graphrag.utils.logger_util import setup_logging

logger = setup_logging()

# ── Model configs: (model_name, label_mapping) ────────────────────────
# JNLPBA model: genes, proteins, RNA/DNA, cell types/lines
_JNLPBA_MODEL = "raynardj/ner-gene-dna-rna-jnlpba-pubmed"
_JNLPBA_MAPPING: dict[str, str] = {
    "protein": "Protein",
    "DNA": "Gene",
    "RNA": "Gene",
    "cell_type": "CellType",
    "cell_line": "CellType",
}

# d4data model: diseases, medications, anatomy, procedures, organisms
_D4DATA_MODEL = "d4data/biomedical-ner-all"
_D4DATA_MAPPING: dict[str, str] = {
    "Disease_disorder": "Disease",
    "Medication": "Drug",
    "Biological_structure": "AnatomicalStructure",
    "Biological_attribute": "BiologicalProcess",
    "Diagnostic_procedure": "Technique",
    "Therapeutic_procedure": "Technique",
    "Sign_symptom": "Disease",
    "Lab_value": "Technique",
}

_DEFAULT_MODELS: list[tuple[str, dict[str, str]]] = [
    (_JNLPBA_MODEL, _JNLPBA_MAPPING),
    (_D4DATA_MODEL, _D4DATA_MAPPING),
]


class HuggingFaceNERExtractor:
    """Extract biomedical entities using HuggingFace token classification models.

    Runs multiple complementary NER models and merges results.
    Uses the transformers pipeline API with aggregation_strategy="simple"
    to merge BIO-tagged tokens into entity spans.
    """

    def __init__(
        self,
        models: list[tuple[str, dict[str, str]]] | None = None,
        confidence_threshold: float = 0.5,
        context_window: int = 50,
    ) -> None:
        self.model_configs = models or _DEFAULT_MODELS
        self.confidence_threshold = confidence_threshold
        self.context_window = context_window
        self._pipelines: dict[str, Any] = {}
        self._lock = threading.Lock()

    def _load_pipeline(self, model_name: str) -> Any:
        """Lazy-load a HuggingFace NER pipeline (thread-safe)."""
        if model_name not in self._pipelines:
            with self._lock:
                if model_name not in self._pipelines:
                    from transformers import pipeline

                    self._pipelines[model_name] = pipeline(
                        "ner",
                        model=model_name,
                        aggregation_strategy="simple",
                    )
                    logger.info(f"Loaded HuggingFace NER model: {model_name}")
        return self._pipelines[model_name]

    async def extract(self, text: str) -> ExtractionResult:
        """Extract entities from text using HuggingFace NER models."""
        return await asyncio.to_thread(self._extract_sync, text)

    def _extract_sync(self, text: str) -> ExtractionResult:
        """Run NER across all configured models."""
        all_entities: list[ExtractedEntity] = []

        for model_name, label_mapping in self.model_configs:
            try:
                pipe = self._load_pipeline(model_name)
                # Truncate to avoid exceeding model max length
                predictions = pipe(text[:2000])
            except Exception as e:
                logger.warning(f"HuggingFace NER model {model_name} failed: {e}")
                continue

            short_name = model_name.split("/")[-1]

            for pred in predictions:
                raw_label = pred["entity_group"]
                # Strip B-/I- prefix if still present after aggregation
                label = raw_label.lstrip("B-").lstrip("I-")

                entity_type = label_mapping.get(label)
                if not entity_type:
                    continue

                score = float(pred["score"])
                if score < self.confidence_threshold:
                    continue

                start = pred.get("start", 0)
                end = pred.get("end", 0)

                # Extract context around entity
                ctx_start = max(0, start - self.context_window)
                ctx_end = min(len(text), end + self.context_window)
                context = text[ctx_start:ctx_end]

                all_entities.append(
                    ExtractedEntity(
                        name=pred["word"],
                        type=entity_type,
                        confidence=score,
                        start_pos=start,
                        end_pos=end,
                        context=context,
                        extractor=f"hf:{short_name}",
                    )
                )

        return ExtractionResult(entities=all_entities, source_text=text)
