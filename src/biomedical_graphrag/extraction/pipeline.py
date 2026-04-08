"""Multi-stage extraction pipeline with configurable merge strategies."""

from __future__ import annotations

import asyncio
from enum import Enum

from biomedical_graphrag.extraction.base import (
    EntityExtractor,
    ExtractedEntity,
    ExtractedRelation,
    ExtractionResult,
)
from biomedical_graphrag.utils.logger_util import setup_logging

logger = setup_logging()


class MergeStrategy(Enum):
    """How to merge results from multiple extraction stages."""

    UNION = "union"
    CONFIDENCE = "confidence"
    CASCADE = "cascade"


class ExtractionPipeline:
    """Run multiple extractors sequentially and merge results."""

    def __init__(
        self,
        stages: list[EntityExtractor],
        merge_strategy: MergeStrategy = MergeStrategy.CONFIDENCE,
        confidence_threshold: float = 0.3,
        fallback_on_error: bool = True,
    ) -> None:
        self.stages = stages
        self.merge_strategy = merge_strategy
        self.confidence_threshold = confidence_threshold
        self.fallback_on_error = fallback_on_error

    async def extract(self, text: str) -> ExtractionResult:
        """Run all stages and merge results."""
        stage_results: list[ExtractionResult] = []

        for stage in self.stages:
            stage_name = type(stage).__name__
            try:
                result = await stage.extract(text)
                logger.debug(
                    f"  {stage_name}: {result.entity_count} entities, "
                    f"{result.relation_count} relations"
                )
                stage_results.append(result)
            except Exception as e:
                if self.fallback_on_error:
                    logger.warning(f"  {stage_name} failed, continuing: {e}")
                else:
                    raise

        if not stage_results:
            return ExtractionResult(source_text=text)

        merged = merge_extraction_results(stage_results, self.merge_strategy)
        merged.source_text = text
        return merged.filter_by_confidence(self.confidence_threshold).filter_invalid()

    async def extract_batch(
        self,
        texts: list[str],
        concurrency: int = 10,
    ) -> list[ExtractionResult]:
        """Process multiple texts with concurrency control."""
        semaphore = asyncio.Semaphore(concurrency)

        async def _extract_one(text: str) -> ExtractionResult:
            async with semaphore:
                return await self.extract(text)

        return await asyncio.gather(*[_extract_one(t) for t in texts])


def merge_extraction_results(
    results: list[ExtractionResult],
    strategy: MergeStrategy,
) -> ExtractionResult:
    """Merge multiple ExtractionResults using the given strategy."""
    if len(results) == 1:
        return results[0]

    if strategy == MergeStrategy.UNION:
        return _merge_union(results)
    elif strategy == MergeStrategy.CONFIDENCE:
        return _merge_confidence(results)
    elif strategy == MergeStrategy.CASCADE:
        return _merge_cascade(results)
    else:
        return _merge_confidence(results)


def _merge_union(results: list[ExtractionResult]) -> ExtractionResult:
    """Keep all unique entities; for duplicates, prefer higher confidence."""
    best: dict[tuple[str, str], ExtractedEntity] = {}
    for result in results:
        for entity in result.entities:
            key = entity.dedup_key
            if key not in best or entity.confidence > best[key].confidence:
                best[key] = entity

    all_relations = _merge_relations(results)
    return ExtractionResult(entities=list(best.values()), relations=all_relations)


def _merge_confidence(results: list[ExtractionResult]) -> ExtractionResult:
    """Keep highest confidence version of each unique entity."""
    return _merge_union(results)  # Same logic: dedup by key, keep highest confidence


def _merge_cascade(results: list[ExtractionResult]) -> ExtractionResult:
    """First extractor as base, fill gaps from later stages."""
    if not results:
        return ExtractionResult()

    best: dict[tuple[str, str], ExtractedEntity] = {}
    for result in results:
        for entity in result.entities:
            key = entity.dedup_key
            if key not in best:  # Only add if not already present
                best[key] = entity

    all_relations = _merge_relations(results)
    return ExtractionResult(entities=list(best.values()), relations=all_relations)


def _merge_relations(results: list[ExtractionResult]) -> list[ExtractedRelation]:
    """Deduplicate relations by (source, target, type), keep highest confidence."""
    best: dict[tuple[str, str, str], ExtractedRelation] = {}
    for result in results:
        for rel in result.relations:
            key = (rel.source.strip().lower(), rel.target.strip().lower(), rel.relation_type)
            if key not in best or rel.confidence > best[key].confidence:
                best[key] = rel
    return list(best.values())
