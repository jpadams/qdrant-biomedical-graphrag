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

    # Step 1: Get all ExtractedEntity normalized names from Neo4j
    logger.info("Fetching ExtractedEntity names from Neo4j...")
    with driver.session() as session:
        result = session.run(
            "MATCH (e:ExtractedEntity) RETURN e.normalized_name AS name"
        )
        entity_names = {record["name"] for record in result}
    logger.info(f"Found {len(entity_names)} unique ExtractedEntity names.")

    # Step 2: Build match pairs — (mesh_ui, entity_normalized_name)
    logger.info("Matching MeSH entry terms to extracted entities...")
    match_pairs: list[dict[str, str]] = []
    matched_descriptors = 0

    for desc in descriptors:
        all_terms = [desc.name] + desc.entry_terms
        desc_matched = False
        for term in all_terms:
            normalized = term.strip().lower()
            if normalized in entity_names:
                match_pairs.append({"ui": desc.ui, "entity_name": normalized})
                desc_matched = True
        if desc_matched:
            matched_descriptors += 1

    logger.info(
        f"Found {len(match_pairs)} term matches across {matched_descriptors} descriptors."
    )

    if not match_pairs:
        logger.warning("No matches found between MeSH terms and extracted entities.")
        return {"match_pairs": 0, "matched_descriptors": 0, "relationships_created": 0}

    # Step 3: Batch create SYNONYM_OF relationships
    logger.info("Creating SYNONYM_OF relationships...")
    total_created = 0
    with driver.session() as session:
        for i in range(0, len(match_pairs), batch_size):
            chunk = match_pairs[i : i + batch_size]
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
            total_created += result.single()["created"]

    logger.info(f"✅ Created {total_created} SYNONYM_OF relationships.")

    # Step 4: Store entry terms on MeshTerm nodes for reference
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
        "match_pairs": len(match_pairs),
        "matched_descriptors": matched_descriptors,
        "relationships_created": total_created,
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
