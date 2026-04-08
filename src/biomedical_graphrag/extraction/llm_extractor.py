"""OpenAI-based biomedical entity and relationship extraction (minimal mode)."""

from __future__ import annotations

import asyncio
import json

from openai import OpenAI

from biomedical_graphrag.config import settings
from biomedical_graphrag.extraction.base import (
    ExtractedEntity,
    ExtractedRelation,
    ExtractionResult,
)
from biomedical_graphrag.extraction.schema import (
    BIOMEDICAL_ENTITY_TYPES,
    BIOMEDICAL_RELATION_TYPES,
    LLM_EXTRACTION_SYSTEM_PROMPT,
    LLM_EXTRACTION_USER_PROMPT,
)
from biomedical_graphrag.utils.logger_util import setup_logging

logger = setup_logging()


class LLMEntityExtractor:
    """Extract biomedical entities and relationships using OpenAI structured output."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.model = model or settings.extraction.llm_model
        key = api_key or settings.openai.api_key.get_secret_value()
        self._client = OpenAI(api_key=key)

    async def extract(self, text: str) -> ExtractionResult:
        """Extract entities and relationships from text via LLM."""
        return await asyncio.to_thread(self._extract_sync, text)

    def _extract_sync(self, text: str) -> ExtractionResult:
        """Synchronous extraction (run in thread pool)."""
        user_prompt = LLM_EXTRACTION_USER_PROMPT.format(abstract=text)

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": LLM_EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )

            content = response.choices[0].message.content or "{}"
            data = json.loads(content)
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"LLM extraction failed: {e}")
            return ExtractionResult(source_text=text)

        entities = _parse_entities(data.get("entities", []))
        relations = _parse_relations(data.get("relations", []))

        return ExtractionResult(
            entities=entities,
            relations=relations,
            source_text=text,
        )


def _parse_entities(raw: list[dict]) -> list[ExtractedEntity]:
    """Parse LLM JSON output into ExtractedEntity objects."""
    entities: list[ExtractedEntity] = []
    for item in raw:
        name = item.get("name", "").strip()
        entity_type = item.get("type", "").strip()
        confidence = float(item.get("confidence", 0.5))

        if not name or not entity_type:
            continue

        # Map to closest valid type if not exact match
        if entity_type not in BIOMEDICAL_ENTITY_TYPES:
            entity_type = _fuzzy_map_type(entity_type)
            if not entity_type:
                continue

        entities.append(
            ExtractedEntity(
                name=name,
                type=entity_type,
                confidence=confidence,
                extractor="llm",
            )
        )
    return entities


def _parse_relations(raw: list[dict]) -> list[ExtractedRelation]:
    """Parse LLM JSON output into ExtractedRelation objects."""
    relations: list[ExtractedRelation] = []
    for item in raw:
        source = item.get("source", "").strip()
        target = item.get("target", "").strip()
        rel_type = item.get("relation_type", "").strip()
        confidence = float(item.get("confidence", 0.5))

        if not source or not target or not rel_type:
            continue

        # Normalize relation type
        rel_upper = rel_type.upper().replace(" ", "_")
        if rel_upper not in BIOMEDICAL_RELATION_TYPES:
            rel_upper = "ASSOCIATED_WITH"

        relations.append(
            ExtractedRelation(
                source=source,
                target=target,
                relation_type=rel_upper,
                confidence=confidence,
                extractor="llm",
            )
        )
    return relations


# Fuzzy type mapping for common LLM output variations
_TYPE_ALIASES: dict[str, str] = {
    "gene": "Gene",
    "protein": "Protein",
    "disease": "Disease",
    "drug": "Drug",
    "chemical": "Drug",
    "compound": "Drug",
    "medication": "Drug",
    "cell_type": "CellType",
    "celltype": "CellType",
    "cell type": "CellType",
    "cell line": "CellType",
    "cell": "CellType",
    "organism": "Organism",
    "species": "Organism",
    "technique": "Technique",
    "method": "Technique",
    "technology": "Technique",
    "tool": "Technique",
    "biological_process": "BiologicalProcess",
    "biologicalprocess": "BiologicalProcess",
    "biological process": "BiologicalProcess",
    "pathway": "BiologicalProcess",
    "process": "BiologicalProcess",
    "mechanism": "BiologicalProcess",
    "anatomical_structure": "AnatomicalStructure",
    "anatomicalstructure": "AnatomicalStructure",
    "anatomical structure": "AnatomicalStructure",
    "anatomy": "AnatomicalStructure",
    "tissue": "AnatomicalStructure",
    "organ": "AnatomicalStructure",
}


def _fuzzy_map_type(raw_type: str) -> str | None:
    """Map LLM output type to our canonical types."""
    return _TYPE_ALIASES.get(raw_type.lower().strip())
