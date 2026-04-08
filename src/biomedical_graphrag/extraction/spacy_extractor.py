"""scispaCy-based biomedical NER extractor (maximal mode)."""

from __future__ import annotations

import asyncio
from typing import Any

from biomedical_graphrag.extraction.base import ExtractedEntity, ExtractionResult
from biomedical_graphrag.extraction.schema import (
    SCISPACY_BC5CDR_MAPPING,
    SCISPACY_CRAFT_MAPPING,
)
from biomedical_graphrag.utils.logger_util import setup_logging

logger = setup_logging()

# Model name → label mapping
_MODEL_MAPPINGS: dict[str, dict[str, str]] = {
    "en_ner_bc5cdr_md": SCISPACY_BC5CDR_MAPPING,
    "en_ner_craft_md": SCISPACY_CRAFT_MAPPING,
}

_DEFAULT_MODELS = ["en_ner_bc5cdr_md", "en_ner_craft_md"]


class ScispaCyExtractor:
    """Extract biomedical entities using scispaCy NER models.

    Runs two models by default:
    - en_ner_bc5cdr_md: Diseases and chemicals
    - en_ner_craft_md: Proteins, genes, organisms, cell types
    """

    def __init__(
        self,
        models: list[str] | None = None,
        default_confidence: float = 0.85,
        context_window: int = 50,
    ) -> None:
        self.model_names = models or _DEFAULT_MODELS
        self.default_confidence = default_confidence
        self.context_window = context_window
        self._loaded_models: dict[str, Any] = {}

    def _load_model(self, model_name: str) -> Any:
        """Lazy-load a spaCy model."""
        if model_name not in self._loaded_models:
            import spacy

            self._loaded_models[model_name] = spacy.load(model_name)
            logger.info(f"Loaded scispaCy model: {model_name}")
        return self._loaded_models[model_name]

    async def extract(self, text: str) -> ExtractionResult:
        """Extract entities from text using scispaCy models."""
        return await asyncio.to_thread(self._extract_sync, text)

    def _extract_sync(self, text: str) -> ExtractionResult:
        """Run NER across all configured models."""
        all_entities: list[ExtractedEntity] = []

        for model_name in self.model_names:
            mapping = _MODEL_MAPPINGS.get(model_name, {})
            if not mapping:
                logger.warning(f"No label mapping for model {model_name}, skipping")
                continue

            try:
                nlp = self._load_model(model_name)
                doc = nlp(text)

                for ent in doc.ents:
                    entity_type = mapping.get(ent.label_)
                    if not entity_type:
                        continue

                    # Extract context around entity
                    start = max(0, ent.start_char - self.context_window)
                    end = min(len(text), ent.end_char + self.context_window)
                    context = text[start:end]

                    all_entities.append(
                        ExtractedEntity(
                            name=ent.text,
                            type=entity_type,
                            confidence=self.default_confidence,
                            start_pos=ent.start_char,
                            end_pos=ent.end_char,
                            context=context,
                            extractor=f"scispacy:{model_name}",
                        )
                    )
            except Exception as e:
                logger.warning(f"scispaCy model {model_name} failed: {e}")

        return ExtractionResult(entities=all_entities, source_text=text)
