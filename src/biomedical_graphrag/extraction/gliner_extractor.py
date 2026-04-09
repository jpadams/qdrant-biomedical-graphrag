"""GLiNER-BioMed entity extraction and GLiREL relationship extraction (maximal mode)."""

from __future__ import annotations

import asyncio
import threading
from typing import Any

from biomedical_graphrag.extraction.base import (
    ExtractedEntity,
    ExtractedRelation,
    ExtractionResult,
)
from biomedical_graphrag.extraction.schema import (
    BIOMEDICAL_RELATION_TYPES,
    GLIREL_RELATION_LABELS,
    GLINER_BIOMEDICAL_LABELS,
)
from biomedical_graphrag.utils.logger_util import setup_logging

logger = setup_logging()


class GLiNERBiomedExtractor:
    """Zero-shot biomedical NER using GLiNER-BioMed, with optional GLiREL relations.

    GLiNER uses entity type descriptions to perform zero-shot NER without
    domain-specific training. GLiREL extracts relationships between entities.
    """

    def __init__(
        self,
        model: str = "Ihor/gliner-biomed-large-v1.0",
        threshold: float = 0.4,
        extract_relations: bool = True,
        glirel_model: str = "jackboyla/glirel-large-v0",
        relation_threshold: float = 0.3,
    ) -> None:
        self.model_name = model
        self.threshold = threshold
        self.extract_relations = extract_relations
        self.glirel_model_name = glirel_model
        self.relation_threshold = relation_threshold
        self._gliner_model: Any = None
        self._glirel_model: Any = None
        self._lock = threading.Lock()

    def _load_gliner(self) -> Any:
        """Lazy-load GLiNER model (thread-safe)."""
        if self._gliner_model is None:
            with self._lock:
                if self._gliner_model is None:
                    from gliner import GLiNER

                    self._gliner_model = GLiNER.from_pretrained(self.model_name)
                    logger.info(f"Loaded GLiNER model: {self.model_name}")
        return self._gliner_model

    def _load_glirel(self) -> Any:
        """Lazy-load GLiREL model (thread-safe)."""
        if self._glirel_model is None:
            with self._lock:
                if self._glirel_model is None:
                    try:
                        from glirel import GLiREL

                        # Patch _from_pretrained: glirel 1.2.1 requires proxies/resume_download
                        # kwargs that huggingface_hub >=1.0 no longer passes.
                        _orig_fn = GLiREL._from_pretrained.__func__

                        @classmethod  # type: ignore[misc]
                        def _patched(cls, *args, **kwargs):  # type: ignore[no-untyped-def]
                            kwargs.setdefault("proxies", None)
                            kwargs.setdefault("resume_download", False)
                            return _orig_fn(cls, *args, **kwargs)

                        GLiREL._from_pretrained = _patched  # type: ignore[assignment]

                        self._glirel_model = GLiREL.from_pretrained(self.glirel_model_name)
                        logger.info(f"Loaded GLiREL model: {self.glirel_model_name}")
                    except ImportError:
                        logger.warning("GLiREL not installed, relation extraction disabled")
                        self.extract_relations = False
                    except Exception as e:
                        logger.warning(f"GLiREL model loading failed: {e}")
                        self.extract_relations = False
        return self._glirel_model

    async def extract(self, text: str) -> ExtractionResult:
        """Extract entities (and optionally relations) from text."""
        return await asyncio.to_thread(self._extract_sync, text)

    def _extract_sync(self, text: str) -> ExtractionResult:
        """Synchronous extraction."""
        entities = self._extract_entities(text)
        relations: list[ExtractedRelation] = []

        if self.extract_relations and entities:
            relations = self._extract_relations_sync(text, entities)

        return ExtractionResult(
            entities=entities,
            relations=relations,
            source_text=text,
        )

    def _extract_entities(self, text: str) -> list[ExtractedEntity]:
        """Run GLiNER zero-shot NER."""
        model = self._load_gliner()
        labels = list(GLINER_BIOMEDICAL_LABELS.keys())

        try:
            predictions = model.predict_entities(text, labels, threshold=self.threshold)
        except Exception as e:
            logger.warning(f"GLiNER prediction failed: {e}")
            return []

        entities: list[ExtractedEntity] = []
        for pred in predictions:
            entities.append(
                ExtractedEntity(
                    name=pred["text"],
                    type=pred["label"],
                    confidence=float(pred["score"]),
                    start_pos=pred.get("start"),
                    end_pos=pred.get("end"),
                    extractor="gliner",
                )
            )
        return entities

    @staticmethod
    def _tokenize(text: str) -> list[tuple[str, int, int]]:
        """Tokenize text the same way GLiREL does internally.

        Returns list of (token, char_start, char_end) tuples.
        """
        import re as _re

        return [
            (m.group(), m.start(), m.end())
            for m in _re.finditer(r"\w+(?:[-_]\w+)*|\S", text)
        ]

    def _char_to_token_spans(
        self, text: str, entities: list[ExtractedEntity]
    ) -> list[list[int | str]]:
        """Convert entity character spans to token index spans for GLiREL."""
        tokens = self._tokenize(text)

        # Build char offset -> token index lookup
        char_to_tok: dict[int, int] = {}
        for idx, (_tok, start, end) in enumerate(tokens):
            for c in range(start, end):
                char_to_tok[c] = idx

        ner_spans: list[list[int | str]] = []
        for e in entities:
            if e.start_pos is None or e.end_pos is None:
                continue
            start_tok = char_to_tok.get(e.start_pos)
            end_tok = char_to_tok.get(e.end_pos - 1)
            if start_tok is None or end_tok is None:
                continue
            # GLiREL expects [start_token, end_token, label, text]
            ner_spans.append([start_tok, end_tok, e.type, e.name])

        return ner_spans

    def _extract_relations_sync(
        self, text: str, entities: list[ExtractedEntity]
    ) -> list[ExtractedRelation]:
        """Run GLiREL relation extraction between extracted entities."""
        glirel = self._load_glirel()
        if glirel is None:
            return []

        if len(entities) < 2:
            return []

        ner_spans = self._char_to_token_spans(text, entities)

        if len(ner_spans) < 2:
            return []

        # GLiREL expects labels as a list of natural language strings
        relation_labels = list(GLIREL_RELATION_LABELS.values())

        try:
            # Use threshold=0.0 to see all candidates, then filter ourselves
            all_predictions = glirel.predict_relations(
                text,
                relation_labels,
                threshold=0.0,
                ner=ner_spans,
            )
            if all_predictions:
                scores = [p.get("score", 0) for p in all_predictions]
                logger.debug(
                    f"  GLiREL raw: {len(all_predictions)} candidates, "
                    f"scores: min={min(scores):.3f} max={max(scores):.3f} "
                    f"median={sorted(scores)[len(scores)//2]:.3f}, "
                    f"above {self.relation_threshold}: {sum(1 for s in scores if s >= self.relation_threshold)}"
                )
            predictions = [p for p in all_predictions if p.get("score", 0) >= self.relation_threshold]
        except Exception as e:
            logger.warning(f"GLiREL prediction failed: {e}")
            return []

        # Map natural language labels back to canonical relation types
        label_to_type = {v: k for k, v in GLIREL_RELATION_LABELS.items()}

        relations: list[ExtractedRelation] = []
        for pred in predictions:
            raw_label = pred.get("label", "")
            rel_type = label_to_type.get(raw_label, "ASSOCIATED_WITH")

            head = pred.get("head_text", "")
            tail = pred.get("tail_text", "")
            # head_text/tail_text may be a list of tokens
            if isinstance(head, list):
                head = " ".join(head)
            if isinstance(tail, list):
                tail = " ".join(tail)

            relations.append(
                ExtractedRelation(
                    source=head,
                    target=tail,
                    relation_type=rel_type,
                    confidence=float(pred.get("score", 0.5)),
                    extractor="glirel",
                )
            )
        return relations
