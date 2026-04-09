"""Factory functions for creating extraction pipelines."""

from __future__ import annotations

from biomedical_graphrag.extraction.llm_extractor import LLMEntityExtractor
from biomedical_graphrag.extraction.pipeline import ExtractionPipeline, MergeStrategy
from biomedical_graphrag.utils.logger_util import setup_logging

logger = setup_logging()


def create_minimal_pipeline() -> ExtractionPipeline:
    """OpenAI-only extraction. No extra dependencies required."""
    return ExtractionPipeline(
        stages=[LLMEntityExtractor()],
        merge_strategy=MergeStrategy.CONFIDENCE,
    )


def create_maximal_pipeline() -> ExtractionPipeline:
    """All tools: HuggingFace NER -> GLiNER-BioMed -> LLM fallback.

    Requires optional dependencies: transformers, torch, gliner, glirel.
    Falls back gracefully if any are unavailable.
    """
    stages = []

    # Stage 1: HuggingFace biomedical NER (fast baseline)
    try:
        from biomedical_graphrag.extraction.hf_extractor import HuggingFaceNERExtractor

        stages.append(HuggingFaceNERExtractor())
        logger.info("Maximal pipeline: HuggingFace NER loaded")
    except ImportError:
        logger.warning(
            "transformers not available, skipping HF NER "
            "(pip install transformers torch)"
        )

    # Stage 2: GLiNER-BioMed (zero-shot)
    try:
        from biomedical_graphrag.extraction.gliner_extractor import GLiNERBiomedExtractor

        stages.append(GLiNERBiomedExtractor())
        logger.info("Maximal pipeline: GLiNER-BioMed loaded")
    except ImportError:
        logger.warning("GLiNER not available, skipping (pip install gliner)")

    # Stage 3: LLM fallback (always available)
    stages.append(LLMEntityExtractor())
    logger.info("Maximal pipeline: LLM fallback loaded")

    return ExtractionPipeline(
        stages=stages,
        merge_strategy=MergeStrategy.CONFIDENCE,
        fallback_on_error=True,
    )
