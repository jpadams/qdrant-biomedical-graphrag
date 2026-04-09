"""Unified ingestion: Qdrant embeddings + Neo4j graph + entity extraction in lock-step."""

import argparse
import asyncio
import json
from dataclasses import asdict
from pathlib import Path

from biomedical_graphrag.config import settings
from biomedical_graphrag.domain.dataset import GeneDataset, PaperDataset
from biomedical_graphrag.extraction.base import (
    ExtractedEntity,
    ExtractedRelation,
    ExtractionResult,
)
from biomedical_graphrag.infrastructure.neo4j_db.neo4j_client import AsyncNeo4jClient
from biomedical_graphrag.infrastructure.neo4j_db.neo4j_graph_schema import Neo4jGraphIngestion
from biomedical_graphrag.infrastructure.qdrant_engine.qdrant_vectorstore import AsyncQdrantVectorStore
from biomedical_graphrag.utils.json_util import load_gene_json, load_pubmed_json
from biomedical_graphrag.utils.logger_util import setup_logging

logger = setup_logging()

EXTRACTION_CACHE_DIR = Path("data/.extraction_cache")


def _cache_path(extraction_mode: str) -> Path:
    """Return the cache file path for a given extraction mode."""
    return EXTRACTION_CACHE_DIR / f"extraction_{extraction_mode}.json"


def _save_extraction_cache(
    results: dict[str, ExtractionResult], extraction_mode: str
) -> None:
    """Persist extraction results to disk so later phases can reuse them."""
    EXTRACTION_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(extraction_mode)
    serialized = {
        pmid: {
            "entities": [asdict(e) for e in r.entities],
            "relations": [asdict(rel) for rel in r.relations],
        }
        for pmid, r in results.items()
    }
    path.write_text(json.dumps(serialized))
    total_e = sum(len(v["entities"]) for v in serialized.values())
    total_r = sum(len(v["relations"]) for v in serialized.values())
    logger.info(
        f"Cached extraction results to {path} "
        f"({len(serialized)} papers, {total_e} entities, {total_r} relations)"
    )


def _load_extraction_cache(extraction_mode: str) -> dict[str, ExtractionResult] | None:
    """Load cached extraction results if available."""
    path = _cache_path(extraction_mode)
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text())
        results: dict[str, ExtractionResult] = {}
        for pmid, data in raw.items():
            entities = [ExtractedEntity(**e) for e in data["entities"]]
            relations = [ExtractedRelation(**r) for r in data["relations"]]
            results[pmid] = ExtractionResult(entities=entities, relations=relations)
        total_e = sum(r.entity_count for r in results.values())
        total_r = sum(r.relation_count for r in results.values())
        logger.info(
            f"Loaded cached extraction from {path} "
            f"({len(results)} papers, {total_e} entities, {total_r} relations)"
        )
        return results
    except Exception as e:
        logger.warning(f"Failed to load extraction cache: {e}")
        return None


def _build_pmid_to_point_id(pubmed_data: dict) -> dict[str, int]:
    """Build deterministic PMID -> Qdrant point ID mapping.

    The Qdrant vectorstore uses int(pmid) as the point ID,
    and skips papers without an abstract or pmid.
    """
    mapping: dict[str, int] = {}
    for paper in pubmed_data.get("papers", []):
        pmid = paper.get("pmid")
        abstract = paper.get("abstract")
        if pmid and abstract:
            mapping[pmid] = int(pmid)
    return mapping


async def run_extraction(
    pubmed_data: dict,
    extraction_mode: str,
) -> dict[str, ExtractionResult]:
    """Run entity extraction on all paper abstracts.

    Args:
        pubmed_data: Raw pubmed dataset dict.
        extraction_mode: "minimal", "gliner", or "maximal".

    Returns:
        Mapping of PMID -> ExtractionResult.
    """
    from biomedical_graphrag.extraction.factory import (
        create_gliner_pipeline,
        create_maximal_pipeline,
        create_minimal_pipeline,
    )

    if extraction_mode == "maximal":
        pipeline = create_maximal_pipeline()
    elif extraction_mode == "gliner":
        pipeline = create_gliner_pipeline()
    else:
        pipeline = create_minimal_pipeline()

    papers = pubmed_data.get("papers", [])
    # Build (pmid, abstract) pairs for papers that have both
    items: list[tuple[str, str]] = []
    for paper in papers:
        pmid = paper.get("pmid")
        abstract = paper.get("abstract")
        if pmid and abstract:
            items.append((pmid, abstract))

    logger.info(f"Extracting entities from {len(items)} abstracts ({extraction_mode} mode)...")

    abstracts = [abstract for _, abstract in items]
    results = await pipeline.extract_batch(
        abstracts,
        concurrency=settings.extraction.batch_concurrency,
    )

    extraction_results: dict[str, ExtractionResult] = {}
    total_entities = 0
    total_relations = 0
    for (pmid, _), result in zip(items, results, strict=True):
        extraction_results[pmid] = result
        total_entities += result.entity_count
        total_relations += result.relation_count

    logger.info(
        f"Extraction complete: {total_entities} entities, "
        f"{total_relations} relations from {len(items)} papers"
    )
    return extraction_results


async def run_unified_ingestion(
    recreate: bool = False,
    only_new: bool = True,
    extraction_mode: str | None = None,
    no_cache: bool = False,
) -> None:
    """Run Qdrant and Neo4j ingestion in lock-step from shared data.

    Args:
        recreate: If True, recreate the Qdrant collection (deletes existing data).
        only_new: If True, only ingest papers not already in Qdrant.
        extraction_mode: "none", "minimal", or "maximal". Defaults to config setting.
        no_cache: If True, ignore cached extraction results and re-run extraction.
    """
    if extraction_mode is None:
        extraction_mode = settings.extraction.mode

    # ── Load data once ──────────────────────────────────────────────
    logger.info("Loading datasets...")
    pubmed_data = load_pubmed_json()
    gene_data = load_gene_json()
    total_papers = len(pubmed_data.get("papers", []))
    total_genes = len(gene_data.get("genes", []))
    logger.info(f"Loaded {total_papers} papers and {total_genes} genes")

    # ── Phase 0: Entity extraction (with cache) ─────────────────────
    extraction_results: dict[str, ExtractionResult] | None = None
    if extraction_mode != "none":
        logger.info(f"=== Phase 0: Entity extraction ({extraction_mode}) ===")
        extraction_results = None if no_cache else _load_extraction_cache(extraction_mode)
        if extraction_results is None:
            extraction_results = await run_extraction(pubmed_data, extraction_mode)
            _save_extraction_cache(extraction_results, extraction_mode)

    # ── Phase 1: Qdrant ─────────────────────────────────────────────
    logger.info("=== Phase 1: Qdrant ingestion ===")
    vector_store = AsyncQdrantVectorStore()
    try:
        if await vector_store.client.collection_exists(vector_store.collection_name):
            if recreate:
                logger.info("Recreating Qdrant collection...")
                await vector_store.delete_collection()
                await vector_store.create_collection()
        else:
            logger.info(f"Collection '{vector_store.collection_name}' does not exist. Creating...")
            await vector_store.create_collection()

        await vector_store.upsert_points(pubmed_data, gene_data, only_new=only_new)
        logger.info("Qdrant ingestion complete.")
    finally:
        await vector_store.close()

    # ── Build cross-reference mapping ───────────────────────────────
    pmid_to_point_id = _build_pmid_to_point_id(pubmed_data)
    collection_name = vector_store.collection_name
    logger.info(f"Built {len(pmid_to_point_id)} PMID -> Qdrant point ID mappings")

    # ── Phase 2: Neo4j ──────────────────────────────────────────────
    logger.info("=== Phase 2: Neo4j ingestion ===")
    dataset = PaperDataset(**pubmed_data)

    gene_dataset: GeneDataset | None = None
    if gene_data.get("genes"):
        gene_dataset = GeneDataset(**gene_data)

    client = await AsyncNeo4jClient.create()
    try:
        ingestion = Neo4jGraphIngestion(client)

        await ingestion.ingest_paper_dataset(
            dataset,
            qdrant_point_ids=pmid_to_point_id,
            qdrant_collection=collection_name,
        )

        if gene_dataset is not None:
            await ingestion.ingest_genes(gene_dataset)

        # ── Phase 3: Ingest extracted entities ──────────────────────
        if extraction_results is not None:
            logger.info("=== Phase 3: Extracted entity ingestion ===")
            await ingestion.ingest_extracted_entities(extraction_results)

        logger.info("Neo4j ingestion complete.")
    finally:
        await client.close()

    # ── Summary ─────────────────────────────────────────────────────
    logger.info("=== Unified ingestion complete ===")
    logger.info(f"  Qdrant: {len(pmid_to_point_id)} papers in '{collection_name}'")
    logger.info(
        f"  Neo4j:  {len(dataset.papers)} papers, {dataset.metadata.total_authors} authors, "
        f"{dataset.metadata.total_mesh_terms} MeSH terms"
    )
    if gene_dataset is not None:
        logger.info(f"  Neo4j:  {len(gene_dataset.genes)} genes linked to papers")
    logger.info("  Cross-ref: qdrant_point_id + qdrant_collection set on Paper nodes")
    if extraction_results is not None:
        total_e = sum(r.entity_count for r in extraction_results.values())
        total_r = sum(r.relation_count for r in extraction_results.values())
        logger.info(f"  Extraction: {total_e} entities, {total_r} relations ({extraction_mode} mode)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Qdrant + Neo4j ingestion")
    parser.add_argument(
        "--extraction-mode",
        choices=["none", "minimal", "gliner", "maximal"],
        default=None,
        help="Entity extraction mode (default: from config/env EXTRACTION__MODE)",
    )
    parser.add_argument("--recreate", action="store_true", help="Recreate Qdrant collection")
    parser.add_argument("--no-cache", action="store_true", help="Ignore cached extraction results")
    args = parser.parse_args()

    asyncio.run(
        run_unified_ingestion(
            recreate=args.recreate,
            only_new=not args.recreate,
            extraction_mode=args.extraction_mode,
            no_cache=args.no_cache,
        )
    )
