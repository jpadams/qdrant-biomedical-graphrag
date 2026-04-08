"""Core data model for biomedical entity and relationship extraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

ENTITY_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "this", "that",
    "these", "those", "it", "its", "they", "them", "their", "we", "our",
    "he", "she", "his", "her", "not", "no", "nor", "also", "more", "most",
    "very", "just", "only", "both", "each", "all", "any", "some", "such",
    "than", "then", "so", "as", "if", "when", "where", "how", "what",
    "which", "who", "whom", "whose", "here", "there", "study", "studies",
    "result", "results", "method", "methods", "however", "although",
    "therefore", "thus", "respectively", "approximately", "significantly",
})


@dataclass
class ExtractedEntity:
    """An entity extracted from text."""

    name: str
    type: str
    confidence: float
    start_pos: int | None = None
    end_pos: int | None = None
    context: str = ""
    extractor: str = ""
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def normalized_name(self) -> str:
        """Lowercase, trimmed entity name for deduplication."""
        return self.name.strip().lower()

    @property
    def dedup_key(self) -> tuple[str, str]:
        """Key for deduplication: (normalized_name, type)."""
        return (self.normalized_name, self.type)


def is_valid_entity(entity: ExtractedEntity) -> bool:
    """Filter out invalid entities (too short, stopwords, purely numeric)."""
    name = entity.name.strip()
    if len(name) < 2:
        return False
    if name.lower() in ENTITY_STOPWORDS:
        return False
    return not name.replace(".", "").replace(",", "").replace("-", "").isdigit()


@dataclass
class ExtractedRelation:
    """A relationship extracted between two entities."""

    source: str
    target: str
    relation_type: str
    confidence: float
    extractor: str = ""


@dataclass
class ExtractionResult:
    """Result of extraction from a single text."""

    entities: list[ExtractedEntity] = field(default_factory=list)
    relations: list[ExtractedRelation] = field(default_factory=list)
    source_text: str | None = None

    @property
    def entity_count(self) -> int:
        """Number of extracted entities."""
        return len(self.entities)

    @property
    def relation_count(self) -> int:
        """Number of extracted relations."""
        return len(self.relations)

    def filter_by_confidence(self, threshold: float) -> ExtractionResult:
        """Return a new result with only entities/relations above threshold."""
        return ExtractionResult(
            entities=[e for e in self.entities if e.confidence >= threshold],
            relations=[r for r in self.relations if r.confidence >= threshold],
            source_text=self.source_text,
        )

    def filter_invalid(self) -> ExtractionResult:
        """Remove stopword and junk entities."""
        valid_entities = [e for e in self.entities if is_valid_entity(e)]
        valid_names = {e.normalized_name for e in valid_entities}
        valid_relations = [
            r for r in self.relations
            if r.source.strip().lower() in valid_names and r.target.strip().lower() in valid_names
        ]
        return ExtractionResult(
            entities=valid_entities,
            relations=valid_relations,
            source_text=self.source_text,
        )

    def entities_by_type(self) -> dict[str, list[ExtractedEntity]]:
        """Group entities by type."""
        result: dict[str, list[ExtractedEntity]] = {}
        for entity in self.entities:
            result.setdefault(entity.type, []).append(entity)
        return result


@runtime_checkable
class EntityExtractor(Protocol):
    """Protocol for entity extractors."""

    async def extract(self, text: str) -> ExtractionResult:
        """Extract entities and relations from text."""
        ...
