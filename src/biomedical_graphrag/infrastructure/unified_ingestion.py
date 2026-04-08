"""Unified ingestion: Qdrant embeddings + Neo4j graph in lock-step."""

import asyncio

from biomedical_graphrag.domain.dataset import GeneDataset, PaperDataset
from biomedical_graphrag.infrastructure.neo4j_db.neo4j_client import AsyncNeo4jClient
from biomedical_graphrag.infrastructure.neo4j_db.neo4j_graph_schema import Neo4jGraphIngestion
from biomedical_graphrag.infrastructure.qdrant_engine.qdrant_vectorstore import AsyncQdrantVectorStore
from biomedical_graphrag.utils.json_util import load_gene_json, load_pubmed_json
from biomedical_graphrag.utils.logger_util import setup_logging

logger = setup_logging()


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


async def run_unified_ingestion(recreate: bool = False, only_new: bool = True) -> None:
    """Run Qdrant and Neo4j ingestion in lock-step from shared data.

    Args:
        recreate: If True, recreate the Qdrant collection (deletes existing data).
        only_new: If True, only ingest papers not already in Qdrant.
    """
    # ── Load data once ──────────────────────────────────────────────
    logger.info("Loading datasets...")
    pubmed_data = load_pubmed_json()
    gene_data = load_gene_json()
    total_papers = len(pubmed_data.get("papers", []))
    total_genes = len(gene_data.get("genes", []))
    logger.info(f"Loaded {total_papers} papers and {total_genes} genes")

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

        logger.info("Neo4j ingestion complete.")
    finally:
        await client.close()

    # ── Summary ─────────────────────────────────────────────────────
    logger.info("=== Unified ingestion complete ===")
    logger.info(f"  Qdrant: {len(pmid_to_point_id)} papers in '{collection_name}'")
    logger.info(f"  Neo4j:  {len(dataset.papers)} papers, {dataset.metadata.total_authors} authors, "
                f"{dataset.metadata.total_mesh_terms} MeSH terms")
    if gene_dataset is not None:
        logger.info(f"  Neo4j:  {len(gene_dataset.genes)} genes linked to papers")
    logger.info("  Cross-ref: qdrant_point_id + qdrant_collection set on Paper nodes")


if __name__ == "__main__":
    asyncio.run(run_unified_ingestion(recreate=True, only_new=False))
