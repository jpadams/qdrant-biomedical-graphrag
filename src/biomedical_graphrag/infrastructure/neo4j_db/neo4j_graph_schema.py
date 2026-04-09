"""High-performance async Neo4j graph ingestion for biomedical papers and genes."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from biomedical_graphrag.domain.dataset import GeneDataset, PaperDataset
from biomedical_graphrag.domain.paper import Paper
from biomedical_graphrag.infrastructure.neo4j_db.neo4j_client import AsyncNeo4jClient
from biomedical_graphrag.utils.logger_util import setup_logging

if TYPE_CHECKING:
    from biomedical_graphrag.extraction.base import ExtractionResult

logger = setup_logging()


class Neo4jGraphIngestion:
    """Asynchronous, batched ingestion of biomedical papers and genes into Neo4j."""

    def __init__(
        self, client: AsyncNeo4jClient, concurrency_limit: int = 25, batch_size: int = 100
    ) -> None:
        self.client = client
        self.semaphore = asyncio.Semaphore(concurrency_limit)
        self.batch_size = batch_size

    # =====================================================
    # ================ CONSTRAINTS ========================
    # =====================================================
    async def create_constraints(self) -> None:
        """Ensure unique keys for all biomedical node types."""
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Paper) REQUIRE p.pmid IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Author) REQUIRE a.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (i:Institution) REQUIRE i.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (m:MeshTerm) REQUIRE m.ui IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (j:Journal) REQUIRE j.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (g:Gene) REQUIRE g.gene_id IS UNIQUE",
            (
                "CREATE CONSTRAINT IF NOT EXISTS FOR (e:ExtractedEntity) "
                "REQUIRE (e.normalized_name, e.type) IS UNIQUE"
            ),
        ]
        for c in constraints:
            await self.client.create_graph(c)
        logger.info("✅ Constraints verified or created.")

    # =====================================================
    # ================= PAPER INGESTION ===================
    # =====================================================
    async def ingest_paper_dataset(
        self,
        dataset: PaperDataset,
        qdrant_point_ids: dict[str, int] | None = None,
        qdrant_collection: str | None = None,
    ) -> None:
        """Ingest papers, authors, MeSH, and citations."""
        await self.create_constraints()

        logger.info(f"🧾 Ingesting {len(dataset.papers)} papers asynchronously...")

        # --- Batch paper nodes ---
        for i in range(0, len(dataset.papers), self.batch_size):
            batch = dataset.papers[i : i + self.batch_size]
            await self._create_paper_batch(batch, qdrant_point_ids, qdrant_collection)
            logger.info(f"  → Inserted {i + len(batch)} / {len(dataset.papers)} papers")

        # --- Authors, institutions, MeSH (async concurrent) ---
        tasks = [self._safe_ingest_paper_relationships(paper) for paper in dataset.papers]
        await asyncio.gather(*tasks)
        logger.info("✅ Paper relationships created.")

        # --- Citations ---
        await self.ingest_citations(dataset.citation_network)
        logger.info("✅ Paper ingestion complete.")

    async def _safe_ingest_paper_relationships(self, paper: Paper) -> None:
        """Concurrent-safe ingestion for relationships."""
        async with self.semaphore:
            try:
                if paper.journal:
                    await self._create_journal_relationship(paper.pmid, paper.journal)

                for author in paper.authors:
                    await self._create_author_relationship(paper.pmid, author.name)
                    for affiliation in author.affiliations:
                        await self._create_affiliation_relationship(author.name, affiliation)

                for mesh_term in paper.mesh_terms:
                    await self._create_mesh_term_relationship(
                        paper.pmid,
                        mesh_term.ui,
                        mesh_term.term,
                        mesh_term.major_topic,
                        mesh_term.qualifiers,
                    )
            except Exception as e:
                logger.warning(f"⚠️ Failed to ingest relationships for paper {paper.pmid}: {e}")

    async def _create_paper_batch(
        self,
        papers: list[Paper],
        qdrant_point_ids: dict[str, int] | None = None,
        qdrant_collection: str | None = None,
    ) -> None:
        """Insert papers in batches using UNWIND for speed."""
        query = """
        UNWIND $batch AS row
        MERGE (p:Paper {pmid: row.pmid})
        SET p.title = row.title,
            p.abstract = row.abstract,
            p.publication_date = row.publication_date,
            p.doi = row.doi,
            p.qdrant_point_id = row.qdrant_point_id,
            p.qdrant_collection = row.qdrant_collection
        """
        params = {
            "batch": [
                {
                    "pmid": p.pmid,
                    "title": p.title,
                    "abstract": p.abstract,
                    "publication_date": p.publication_date,
                    "doi": p.doi,
                    "qdrant_point_id": qdrant_point_ids.get(p.pmid) if qdrant_point_ids else None,
                    "qdrant_collection": qdrant_collection,
                }
                for p in papers
            ]
        }
        await self.client.create_graph(query, params)

    async def ingest_citations(self, citation_network: dict[str, Any]) -> None:
        """Create CITES relationships (batched for performance)."""
        all_edges = []
        for pmid, cinfo in citation_network.items():
            refs = getattr(cinfo, "references", [])
            for ref in refs:
                all_edges.append({"citing": pmid, "cited": ref})

        for i in range(0, len(all_edges), self.batch_size * 5):
            batch = all_edges[i : i + self.batch_size * 5]
            query = """
            UNWIND $batch AS edge
            MATCH (p1:Paper {pmid: edge.citing})
            MATCH (p2:Paper {pmid: edge.cited})
            MERGE (p1)-[:CITES]->(p2)
            """
            await self.client.create_graph(query, {"batch": batch})
        logger.info(f"Created {len(all_edges)} citation relationships.")

    # =====================================================
    # ================== GENE INGESTION ===================
    # =====================================================
    async def ingest_genes(self, gene_dataset: GeneDataset) -> None:
        """Ingest genes and link them to papers, then compute co-occurrences."""
        await self.create_constraints()

        genes = getattr(gene_dataset, "genes", [])
        logger.info(f"🧬 Ingesting {len(genes)} genes asynchronously...")

        # --- Batch genes ---
        for i in range(0, len(genes), self.batch_size):
            batch = genes[i : i + self.batch_size]
            await self._create_gene_batch(batch)
            logger.info(f"  → Inserted {i + len(batch)} / {len(genes)} genes")

        # --- Link genes to papers concurrently ---
        tasks = [self._safe_link_gene_to_papers(gene) for gene in genes]
        await asyncio.gather(*tasks)

    async def _create_gene_batch(self, genes: list[Any]) -> None:
        """Insert genes in batches using UNWIND."""
        query = """
        UNWIND $batch AS g
        MERGE (gene:Gene {gene_id: g.gene_id})
        SET gene.name = g.name,
            gene.description = g.description,
            gene.chromosome = g.chromosome,
            gene.map_location = g.map_location,
            gene.organism = g.organism,
            gene.aliases = g.aliases,
            gene.designations = g.designations
        """
        params = {
            "batch": [
                {
                    "gene_id": g.gene_id,
                    "name": g.name,
                    "description": g.description,
                    "chromosome": g.chromosome,
                    "map_location": g.map_location,
                    "organism": g.organism,
                    "aliases": g.aliases,
                    "designations": g.designations,
                }
                for g in genes
            ]
        }
        await self.client.create_graph(query, params)

    async def _safe_link_gene_to_papers(self, gene: Any) -> None:
        """Concurrent-safe creation of Gene–Paper relationships."""
        async with self.semaphore:
            try:
                for pmid in getattr(gene, "linked_pmids", []):
                    if pmid:
                        await self._create_gene_paper_relationship(gene_id=gene.gene_id, pmid=pmid)
            except Exception as e:
                logger.warning(f"⚠️ Failed linking gene {gene.gene_id} → papers: {e}")

    async def _create_gene_paper_relationship(self, *, gene_id: str, pmid: str) -> None:
        query = """
        MERGE (g:Gene {gene_id: $gene_id})
        MERGE (p:Paper {pmid: $pmid})
        MERGE (g)-[:MENTIONED_IN]->(p)
        """
        await self.client.create_graph(query, {"gene_id": gene_id, "pmid": pmid})

    # =====================================================
    # ============= QDRANT CROSS-REFERENCE ===============
    # =====================================================
    async def update_qdrant_references(
        self, pmid_to_point_id: dict[str, int], collection_name: str
    ) -> None:
        """Stamp qdrant_point_id and qdrant_collection onto existing Paper nodes."""
        batch_list = [
            {"pmid": pmid, "point_id": pid, "collection": collection_name}
            for pmid, pid in pmid_to_point_id.items()
        ]
        for i in range(0, len(batch_list), self.batch_size):
            chunk = batch_list[i : i + self.batch_size]
            query = """
            UNWIND $batch AS row
            MATCH (p:Paper {pmid: row.pmid})
            SET p.qdrant_point_id = row.point_id,
                p.qdrant_collection = row.collection
            """
            await self.client.create_graph(query, {"batch": chunk})
        logger.info(f"✅ Stamped Qdrant references on {len(batch_list)} papers.")

    # =====================================================
    # =========== EXTRACTED ENTITY INGESTION ==============
    # =====================================================
    async def ingest_extracted_entities(
        self, extraction_results: dict[str, ExtractionResult]
    ) -> None:
        """Create extracted entity nodes and link them to papers.

        Args:
            extraction_results: Mapping of PMID -> ExtractionResult.
        """
        await self.create_constraints()

        # Collect all unique entities across papers
        unique_entities: dict[tuple[str, str], dict[str, Any]] = {}
        paper_entity_links: list[dict[str, str]] = []
        all_relations: list[dict[str, Any]] = []

        for pmid, result in extraction_results.items():
            for entity in result.entities:
                key = entity.dedup_key
                # Keep highest confidence version
                if key not in unique_entities or entity.confidence > unique_entities[key]["confidence"]:
                    unique_entities[key] = {
                        "normalized_name": entity.normalized_name,
                        "name": entity.name,
                        "type": entity.type,
                        "confidence": entity.confidence,
                        "extractor": entity.extractor,
                    }
                paper_entity_links.append({
                    "pmid": pmid,
                    "normalized_name": entity.normalized_name,
                    "entity_type": entity.type,
                })

            for rel in result.relations:
                all_relations.append({
                    "pmid": pmid,
                    "source_name": rel.source.strip().lower(),
                    "target_name": rel.target.strip().lower(),
                    "relation_type": rel.relation_type,
                    "confidence": rel.confidence,
                    "extractor": rel.extractor,
                })

        # Batch create entity nodes
        entity_batch = list(unique_entities.values())
        for i in range(0, len(entity_batch), self.batch_size):
            chunk = entity_batch[i : i + self.batch_size]
            query = """
            UNWIND $batch AS row
            MERGE (e:ExtractedEntity {normalized_name: row.normalized_name, type: row.type})
            SET e.name = row.name,
                e.confidence = row.confidence,
                e.extractor = row.extractor
            """
            await self.client.create_graph(query, {"batch": chunk})

        logger.info(f"✅ Created {len(entity_batch)} unique extracted entity nodes.")

        # Batch create Paper -> ExtractedEntity relationships
        for i in range(0, len(paper_entity_links), self.batch_size):
            chunk = paper_entity_links[i : i + self.batch_size]
            query = """
            UNWIND $batch AS row
            MATCH (p:Paper {pmid: row.pmid})
            MATCH (e:ExtractedEntity {normalized_name: row.normalized_name, type: row.entity_type})
            MERGE (p)-[:MENTIONED_IN_ABSTRACT]->(e)
            """
            await self.client.create_graph(query, {"batch": chunk})

        logger.info(f"✅ Created {len(paper_entity_links)} paper-entity links.")

        # Batch create inter-entity relationships
        for i in range(0, len(all_relations), self.batch_size):
            chunk = all_relations[i : i + self.batch_size]
            query = """
            UNWIND $batch AS row
            MATCH (s:ExtractedEntity {normalized_name: row.source_name})
            MATCH (t:ExtractedEntity {normalized_name: row.target_name})
            MERGE (s)-[r:RELATED_TO {relation_type: row.relation_type}]->(t)
            SET r.confidence = row.confidence,
                r.extractor = row.extractor
            """
            await self.client.create_graph(query, {"batch": chunk})

        if all_relations:
            logger.info(f"✅ Created {len(all_relations)} inter-entity relationships.")

        # Promote entity types to secondary Neo4j labels
        await self._promote_entity_labels()

        # Add MENTIONED_IN relationships for Gene entities (unifies with curated genes)
        await self._create_gene_mentioned_in_links()

    async def _promote_entity_labels(self, confidence_threshold: float = 0.85) -> None:
        """Add the entity type as a secondary Neo4j label on high-confidence ExtractedEntity nodes.

        E.g., an ExtractedEntity with type='Gene' becomes (:ExtractedEntity:Gene).
        """
        from biomedical_graphrag.extraction.schema import BIOMEDICAL_ENTITY_TYPES

        for entity_type in BIOMEDICAL_ENTITY_TYPES:
            query = f"""
            MATCH (e:ExtractedEntity)
            WHERE e.type = $type AND e.confidence >= $threshold
            SET e:{entity_type}
            RETURN count(e) AS promoted
            """
            result = await self.client.create_graph(
                query, {"type": entity_type, "threshold": confidence_threshold}
            )
            count = result[0]["promoted"] if result else 0
            if count > 0:
                logger.info(f"  Promoted {count} ExtractedEntity nodes to :{entity_type}")

        logger.info("✅ Entity type label promotion complete.")

    async def _create_gene_mentioned_in_links(self) -> None:
        """Create (Gene)-[:MENTIONED_IN]->(Paper) for extracted Gene entities.

        This unifies extracted genes with curated NCBI genes so that
        get_genes_in_same_papers works for both.
        """
        query = """
        MATCH (p:Paper)-[:MENTIONED_IN_ABSTRACT]->(e:ExtractedEntity:Gene)
        MERGE (e)-[:MENTIONED_IN]->(p)
        RETURN count(*) AS created
        """
        result = await self.client.create_graph(query, {})
        count = result[0]["created"] if result else 0
        logger.info(f"✅ Created {count} Gene MENTIONED_IN links (extracted → paper).")

    # =====================================================
    # ============== RELATIONSHIP HELPERS =================
    # =====================================================
    async def _create_journal_relationship(self, pmid: str, journal: str) -> None:
        await self.client.create_graph(
            """
            MERGE (j:Journal {name: $journal})
            MERGE (p:Paper {pmid: $pmid})
            MERGE (p)-[:PUBLISHED_IN]->(j)
            """,
            {"pmid": pmid, "journal": journal},
        )

    async def _create_author_relationship(self, pmid: str, author_name: str) -> None:
        await self.client.create_graph(
            """
            MERGE (a:Author {name: $name})
            MERGE (p:Paper {pmid: $pmid})
            MERGE (a)-[:WROTE]->(p)
            """,
            {"name": author_name, "pmid": pmid},
        )

    async def _create_affiliation_relationship(self, author_name: str, affiliation: str) -> None:
        await self.client.create_graph(
            """
            MERGE (i:Institution {name: $affiliation})
            MERGE (a:Author {name: $name})
            MERGE (a)-[:AFFILIATED_WITH]->(i)
            """,
            {"name": author_name, "affiliation": affiliation},
        )

    async def _create_mesh_term_relationship(
        self, pmid: str, ui: str, term: str, major_topic: bool, qualifiers: list[str]
    ) -> None:
        await self.client.create_graph(
            """
            MERGE (m:MeshTerm {ui: $ui})
            SET m.term = $term
            MERGE (p:Paper {pmid: $pmid})
            MERGE (p)-[r:HAS_MESH_TERM]->(m)
            SET r.major_topic = $major_topic,
                r.qualifiers = $qualifiers
            """,
            {
                "ui": ui,
                "term": term,
                "pmid": pmid,
                "major_topic": major_topic,
                "qualifiers": qualifiers,
            },
        )
