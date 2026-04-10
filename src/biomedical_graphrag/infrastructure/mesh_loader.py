"""Parse MeSH XML descriptors and link entry terms to ExtractedEntity nodes in Neo4j."""

from __future__ import annotations

import gzip
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from biomedical_graphrag.config import settings
from biomedical_graphrag.utils.logger_util import setup_logging

logger = setup_logging()

MESH_XML_PATH = Path("data/desc2026.gz")


@dataclass
class MeshDescriptor:
    ui: str
    name: str
    entry_terms: list[str] = field(default_factory=list)
    tree_numbers: list[str] = field(default_factory=list)


def parse_mesh_xml(path: Path = MESH_XML_PATH) -> list[MeshDescriptor]:
    """Parse MeSH descriptors XML, extracting entry terms (synonyms).

    Args:
        path: Path to desc2026.gz (gzipped XML).

    Returns:
        List of MeshDescriptor objects with entry terms.
    """
    descriptors: list[MeshDescriptor] = []
    opener = gzip.open if path.suffix == ".gz" else open

    logger.info(f"Parsing MeSH XML from {path}...")
    with opener(path, "rb") as f:
        for _event, elem in ET.iterparse(f, events=("end",)):
            if elem.tag != "DescriptorRecord":
                continue

            ui = elem.findtext(".//DescriptorUI") or ""
            name = elem.findtext(".//DescriptorName/String") or ""

            if not ui or not name:
                elem.clear()
                continue

            # Collect all terms across all concepts (these are synonyms)
            entry_terms: list[str] = []
            for concept in elem.findall(".//Concept"):
                for term in concept.findall(".//Term"):
                    term_str = term.findtext("String") or ""
                    if term_str and term_str != name:
                        entry_terms.append(term_str)

            tree_numbers = [tn.text for tn in elem.findall(".//TreeNumber") if tn.text]

            descriptors.append(
                MeshDescriptor(
                    ui=ui,
                    name=name,
                    entry_terms=entry_terms,
                    tree_numbers=tree_numbers,
                )
            )
            elem.clear()

    logger.info(f"Parsed {len(descriptors)} MeSH descriptors.")
    total_terms = sum(len(d.entry_terms) for d in descriptors)
    logger.info(f"Total entry terms (synonyms): {total_terms}")
    return descriptors


def build_synonym_index(descriptors: list[MeshDescriptor]) -> dict[str, list[str]]:
    """Build a lowercase lookup: synonym -> list of MeSH UIs that contain it.

    Includes both the preferred name and all entry terms.
    """
    index: dict[str, list[str]] = {}
    for desc in descriptors:
        for term in [desc.name] + desc.entry_terms:
            key = term.strip().lower()
            if key:
                index.setdefault(key, []).append(desc.ui)
    return index


def link_mesh_to_entities(
    descriptors: list[MeshDescriptor],
    neo4j_uri: str | None = None,
    neo4j_user: str | None = None,
    neo4j_password: str | None = None,
    batch_size: int = 500,
) -> dict[str, int]:
    """Match MeSH entry terms to ExtractedEntity nodes and create SYNONYM_OF relationships.

    For each MeSH descriptor, checks if any of its entry terms match an
    ExtractedEntity.normalized_name. If so, creates:
        (:MeshTerm {ui})-[:SYNONYM_OF]->(:ExtractedEntity {normalized_name})

    Also enriches MeshTerm nodes with entry_terms property for future use.

    Args:
        descriptors: Parsed MeSH descriptors.
        neo4j_uri: Neo4j connection URI (defaults to settings).
        neo4j_user: Neo4j username (defaults to settings).
        neo4j_password: Neo4j password (defaults to settings).
        batch_size: Batch size for Neo4j operations.

    Returns:
        Stats dict with counts of matches and relationships created.
    """
    from neo4j import GraphDatabase

    uri = neo4j_uri or settings.neo4j.uri
    user = neo4j_user or settings.neo4j.username
    password = neo4j_password or settings.neo4j.password.get_secret_value()

    driver = GraphDatabase.driver(uri, auth=(user, password))

    try:
        return _link_mesh_to_entities(driver, descriptors, batch_size)
    finally:
        driver.close()


def _link_mesh_to_entities(
    driver: Any,
    descriptors: list[MeshDescriptor],
    batch_size: int,
) -> dict[str, int]:
    """Internal: create SYNONYM_OF links between MeshTerm and ExtractedEntity nodes."""

    # Step 1: Get all matchable names from Neo4j
    logger.info("Fetching ExtractedEntity names from Neo4j...")
    with driver.session() as session:
        result = session.run(
            "MATCH (e:ExtractedEntity) RETURN e.normalized_name AS name"
        )
        entity_names = {record["name"] for record in result}
    logger.info(f"Found {len(entity_names)} unique ExtractedEntity names.")

    logger.info("Fetching Gene names and aliases from Neo4j...")
    with driver.session() as session:
        result = session.run(
            "MATCH (g:Gene) WHERE NOT g:ExtractedEntity RETURN g.name AS name, g.aliases AS aliases"
        )
        gene_names: dict[str, str] = {}  # lowercase name -> original name
        for record in result:
            name = record["name"]
            if name:
                gene_names[name.strip().lower()] = name
            aliases = record["aliases"] or ""
            for alias in aliases.split(", "):
                alias = alias.strip()
                if alias:
                    gene_names[alias.strip().lower()] = alias
    logger.info(f"Found {len(gene_names)} unique Gene names/aliases.")

    # Step 2: Build match pairs
    logger.info("Matching MeSH entry terms to extracted entities and genes...")
    entity_match_pairs: list[dict[str, str]] = []
    gene_match_pairs: list[dict[str, str]] = []
    matched_descriptors = 0

    for desc in descriptors:
        all_terms = [desc.name] + desc.entry_terms
        desc_matched = False
        for term in all_terms:
            normalized = term.strip().lower()
            if normalized in entity_names:
                entity_match_pairs.append({"ui": desc.ui, "entity_name": normalized})
                desc_matched = True
            if normalized in gene_names:
                gene_match_pairs.append({"ui": desc.ui, "gene_name": gene_names[normalized]})
                desc_matched = True
        if desc_matched:
            matched_descriptors += 1

    logger.info(
        f"Found {len(entity_match_pairs)} entity matches + {len(gene_match_pairs)} gene matches "
        f"across {matched_descriptors} descriptors."
    )

    if not entity_match_pairs and not gene_match_pairs:
        logger.warning("No matches found between MeSH terms and graph nodes.")
        return {"entity_matches": 0, "gene_matches": 0, "matched_descriptors": 0, "relationships_created": 0}

    # Step 3: Batch create SYNONYM_OF relationships for ExtractedEntity
    total_entity_created = 0
    if entity_match_pairs:
        total_batches = (len(entity_match_pairs) + batch_size - 1) // batch_size
        logger.info(f"Creating SYNONYM_OF relationships for {len(entity_match_pairs)} entity matches ({total_batches} batches)...")
        with driver.session() as session:
            for i in range(0, len(entity_match_pairs), batch_size):
                chunk = entity_match_pairs[i : i + batch_size]
                result = session.run(
                    """
                    UNWIND $batch AS row
                    MATCH (m:MeshTerm {ui: row.ui})
                    MATCH (e:ExtractedEntity {normalized_name: row.entity_name})
                    MERGE (m)-[:SYNONYM_OF]->(e)
                    RETURN count(*) AS created
                    """,
                    {"batch": chunk},
                )
                total_entity_created += result.single()["created"]
                batch_num = i // batch_size + 1
                if batch_num % 5 == 0 or batch_num == total_batches:
                    logger.info(f"  → Entity SYNONYM_OF: batch {batch_num}/{total_batches} ({total_entity_created} created)")
        logger.info(f"✅ Created {total_entity_created} MeSH→ExtractedEntity SYNONYM_OF relationships.")

    # Step 4: Batch create SYNONYM_OF relationships for Gene nodes
    total_gene_created = 0
    if gene_match_pairs:
        total_batches = (len(gene_match_pairs) + batch_size - 1) // batch_size
        logger.info(f"Creating SYNONYM_OF relationships for {len(gene_match_pairs)} gene matches ({total_batches} batches)...")
        with driver.session() as session:
            for i in range(0, len(gene_match_pairs), batch_size):
                chunk = gene_match_pairs[i : i + batch_size]
                result = session.run(
                    """
                    UNWIND $batch AS row
                    MATCH (m:MeshTerm {ui: row.ui})
                    MATCH (g:Gene {name: row.gene_name})
                    WHERE NOT g:ExtractedEntity
                    MERGE (m)-[:SYNONYM_OF]->(g)
                    RETURN count(*) AS created
                    """,
                    {"batch": chunk},
                )
                total_gene_created += result.single()["created"]
                batch_num = i // batch_size + 1
                if batch_num % 5 == 0 or batch_num == total_batches:
                    logger.info(f"  → Gene SYNONYM_OF: batch {batch_num}/{total_batches} ({total_gene_created} created)")
        logger.info(f"✅ Created {total_gene_created} MeSH→Gene SYNONYM_OF relationships.")

    total_created = total_entity_created + total_gene_created

    # Step 5: Store entry terms on MeshTerm nodes for reference
    logger.info("Enriching MeshTerm nodes with entry terms...")
    enrichment_batch: list[dict[str, Any]] = []
    # Only enrich MeshTerms that exist in our graph
    with driver.session() as session:
        result = session.run("MATCH (m:MeshTerm) RETURN m.ui AS ui")
        graph_uis = {record["ui"] for record in result}

    for desc in descriptors:
        if desc.ui in graph_uis and desc.entry_terms:
            enrichment_batch.append({
                "ui": desc.ui,
                "entry_terms": desc.entry_terms[:20],  # Cap at 20 to avoid huge properties
            })

    with driver.session() as session:
        for i in range(0, len(enrichment_batch), batch_size):
            chunk = enrichment_batch[i : i + batch_size]
            session.run(
                """
                UNWIND $batch AS row
                MATCH (m:MeshTerm {ui: row.ui})
                SET m.entry_terms = row.entry_terms
                """,
                {"batch": chunk},
            )

    logger.info(f"✅ Enriched {len(enrichment_batch)} MeshTerm nodes with entry terms.")

    stats = {
        "entity_matches": len(entity_match_pairs),
        "gene_matches": len(gene_match_pairs),
        "matched_descriptors": matched_descriptors,
        "entity_relationships_created": total_entity_created,
        "gene_relationships_created": total_gene_created,
        "total_relationships_created": total_created,
        "mesh_terms_enriched": len(enrichment_batch),
    }
    logger.info(f"MeSH linking stats: {stats}")
    return stats


def main() -> None:
    """CLI entry point: parse MeSH XML and link to Neo4j."""
    descriptors = parse_mesh_xml()
    stats = link_mesh_to_entities(descriptors)
    print(f"\nDone. Stats: {stats}")


if __name__ == "__main__":
    main()
