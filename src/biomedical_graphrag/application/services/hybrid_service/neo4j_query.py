from typing import Any

from neo4j import GraphDatabase

from biomedical_graphrag.config import settings
from biomedical_graphrag.utils.logger_util import setup_logging

logger = setup_logging()


class Neo4jGraphQuery:
    """
    Handles querying Neo4j graph using predefined Cypher templates for biomedical enrichment.
    All query templates are static methods in this class.
    """

    def __init__(self) -> None:
        self.uri = settings.neo4j.uri
        self.username = settings.neo4j.username
        self.password = settings.neo4j.password.get_secret_value()
        self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))

    def close(self) -> None:
        """Close the Neo4j driver and release underlying connections."""
        self.driver.close()

    def query(self, cypher: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """
        Execute a raw Cypher query against the graph.
        """
        with self.driver.session() as session:
            result = session.run(cypher, params or {})
            return [dict(record) for record in result]

    def get_schema(self) -> str:
        """
        Get the Neo4j graph schema for biomedical data.
        """
        return """
        Biomedical Graph Schema:

        Nodes:
        - Paper: {pmid, title, abstract, publication_date, doi, qdrant_point_id, qdrant_collection}
        - Author: {name}
        - Institution: {name}
        - MeshTerm: {ui, term}
        - Journal: {name}
        - Gene: {gene_id, name, description, chromosome, map_location, organism, aliases, designations}
          Note: includes both curated NCBI genes AND high-confidence extracted genes
        - ExtractedEntity: {normalized_name, name, type, confidence, extractor}
          Each ExtractedEntity also carries its type as a secondary label:
          :Gene, :Protein, :Disease, :Drug, :CellType, :Organism,
          :Technique, :BiologicalProcess, :AnatomicalStructure

        Relationships:
        - (Author)-[:WROTE]->(Paper)
        - (Author)-[:AFFILIATED_WITH]->(Institution)
        - (Paper)-[:HAS_MESH_TERM {major_topic: boolean, qualifiers: [string]}]->(MeshTerm)
        - (Paper)-[:PUBLISHED_IN]->(Journal)
        - (Paper)-[:CITES]->(Paper)
        - (Gene)-[:MENTIONED_IN]->(Paper)  — curated NCBI genes AND extracted genes
        - (Paper)-[:MENTIONED_IN_ABSTRACT]->(ExtractedEntity)  — NER-extracted entities
        - (MeshTerm)-[:SYNONYM_OF]->(ExtractedEntity)  — MeSH entry terms matched to extracted entities
        """

    def get_collaborators_with_topics(
        self, author_name: str, topics: list[str], require_all: bool = False,
        exclude_pmids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get collaborators for an author filtered by MeSH topics.
        Uses case-insensitive CONTAINS matching for flexibility.
        Note: exclude_pmids is accepted but intentionally not applied here.
        Collaborator networks should be computed across all shared papers,
        including those already retrieved by Qdrant, since the goal is to
        surface people (not new papers).
        """
        if require_all:
            topic_clauses = "\n".join(
                f"MATCH (p)-[:HAS_MESH_TERM]->(m{i}:MeshTerm) WHERE toLower(m{i}.term) CONTAINS toLower($topic_{i})"
                for i in range(len(topics))
            )
            cypher = f"""
                MATCH (a1:Author)-[:WROTE]->(p:Paper)<-[:WROTE]-(a2:Author)
                WHERE toLower(a1.name) CONTAINS toLower($author_name) AND a1 <> a2
                WITH DISTINCT a2, p
                {topic_clauses}
                RETURN DISTINCT a2.name as collaborator, COUNT(DISTINCT p) as papers
                ORDER BY papers DESC
                LIMIT 10
            """
            params: dict[str, Any] = {"author_name": author_name}
            for i, topic in enumerate(topics):
                params[f"topic_{i}"] = topic
        else:
            cypher = """
                MATCH (a1:Author)-[:WROTE]->(p:Paper)<-[:WROTE]-(a2:Author)
                WHERE toLower(a1.name) CONTAINS toLower($author_name) AND a1 <> a2
                WITH DISTINCT a2, p
                MATCH (p)-[:HAS_MESH_TERM]->(m:MeshTerm)
                WHERE ANY(topic IN $topics WHERE toLower(m.term) CONTAINS toLower(topic))
                RETURN DISTINCT a2.name as collaborator,
                       COUNT(DISTINCT p) as papers,
                       COLLECT(DISTINCT m.term)[0..3] as sample_topics
                ORDER BY papers DESC
                LIMIT 10
            """
            params = {"author_name": author_name, "topics": topics}
        return self.query(cypher, params)

    def get_related_papers_by_mesh(
        self, pmid: str, exclude_pmids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get papers related by MeSH terms to a given PMID.
        Optionally excludes papers already retrieved by Qdrant.
        """
        exclude_pmids = exclude_pmids or []
        cypher = """
            MATCH (p1:Paper {pmid: $pmid})-[:HAS_MESH_TERM]->(m)
                  <-[:HAS_MESH_TERM]-(p2:Paper)
            WHERE p1 <> p2 AND NOT p2.pmid IN $exclude_pmids
            WITH p2, COUNT(DISTINCT m) as shared_terms
            RETURN p2.pmid as pmid, p2.title as title, shared_terms
            ORDER BY shared_terms DESC
            LIMIT 10
        """
        return self.query(cypher, {"pmid": pmid, "exclude_pmids": exclude_pmids})

    def get_entities_for_papers(
        self, pmids: list[str], entity_type_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get all extracted entities from specific papers.

        Returns entities grouped by type with the papers they appear in.
        Scoped to the retrieved papers — grounds insights in actual abstracts.
        Filters out low-quality entities (low confidence, generic descriptions,
        HTML artifacts).

        Args:
            pmids: List of PMIDs to query.
            entity_type_filter: Optional type filter (e.g., 'Gene', 'Drug').
        """
        type_clause = (
            "AND e.type = $type_filter" if entity_type_filter else ""
        )
        cypher = f"""
            UNWIND $pmids AS pmid
            MATCH (p:Paper {{pmid: pmid}})-[:MENTIONED_IN_ABSTRACT]->(e:ExtractedEntity)
            WHERE e.confidence >= 0.75
              AND size(e.name) <= 60
              AND NOT e.name CONTAINS '<'
              AND NOT toLower(e.name) ENDS WITH ' genes'
              AND NOT toLower(e.name) ENDS WITH ' proteins'
              AND NOT toLower(e.name) ENDS WITH ' cells'
              AND NOT toLower(e.name) ENDS WITH ' pathways'
              AND NOT toLower(e.name) ENDS WITH ' mechanisms'
              {type_clause}
            WITH e.name AS entity, e.type AS type,
                 COLLECT(DISTINCT p.pmid) AS found_in_pmids,
                 MAX(e.confidence) AS confidence
            RETURN entity, type, confidence,
                   found_in_pmids,
                   SIZE(found_in_pmids) AS paper_count
            ORDER BY type, paper_count DESC, confidence DESC
        """
        params: dict[str, Any] = {"pmids": pmids}
        if entity_type_filter:
            params["type_filter"] = entity_type_filter
        return self.query(cypher, params)

    def get_entity_cooccurrence(
        self,
        entity_name: str,
        synonyms: list[str] | None = None,
        entity_type_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Find entities co-mentioned in the same papers as a target entity.

        Uses ExtractedEntity nodes linked via MENTIONED_IN_ABSTRACT.

        Args:
            entity_name: Name of the target entity (e.g., 'CREBBP', 'glioblastoma').
            synonyms: Optional list of synonyms/abbreviations to also match
                (e.g., ['DLBCL', 'diffuse large B-cell lymphoma'] for 'non-Hodgkin lymphoma').
            entity_type_filter: Optional type to filter results (e.g., 'Gene', 'Drug').
        """
        type_clause = (
            "AND e2.type = $type_filter" if entity_type_filter else ""
        )
        cypher = f"""
            MATCH (e1:ExtractedEntity)<-[:MENTIONED_IN_ABSTRACT]-(p:Paper)
                  -[:MENTIONED_IN_ABSTRACT]->(e2:ExtractedEntity)
            WHERE ANY(term IN $search_terms
                      WHERE toLower(e1.name) = toLower(term)
                         OR toLower(e1.name) CONTAINS toLower(term))
              AND e1 <> e2
              AND e2.confidence >= 0.75
              AND size(e2.name) <= 60
              AND NOT e2.name CONTAINS '<'
              AND NOT toLower(e2.name) ENDS WITH ' genes'
              AND NOT toLower(e2.name) ENDS WITH ' proteins'
              AND NOT toLower(e2.name) ENDS WITH ' cells'
              AND NOT toLower(e2.name) ENDS WITH ' pathways'
              AND NOT toLower(e2.name) ENDS WITH ' mechanisms'
              {type_clause}
            RETURN e2.name AS entity, e2.type AS type,
                   COUNT(DISTINCT p) AS shared_papers,
                   COLLECT(DISTINCT p.pmid)[..5] AS example_pmids
            ORDER BY shared_papers DESC
            LIMIT 15
        """
        search_terms = [entity_name] + (synonyms or [])
        params: dict[str, Any] = {"search_terms": search_terms}
        if entity_type_filter:
            params["type_filter"] = entity_type_filter
        return self.query(cypher, params)

    def get_genes_in_same_papers(
        self, target_gene: str, mesh_filter: str | None = None
    ) -> list[dict[str, Any]]:
        """
        Find genes co-mentioned in the same papers as the target gene.
        Optionally filter by MeSH term substring (e.g., 'cancer', 'HIV').

        Examples:
            - "Which genes are mentioned in the same papers as gag?"
            - "Which genes co-occur with CCR5 in HIV-related papers?"
        """
        cypher = """
            MATCH (g:Gene)
            WHERE toLower(g.name) CONTAINS toLower($target_gene)
            OR toLower(g.aliases) CONTAINS toLower($target_gene)
            MATCH (g)-[:MENTIONED_IN]->(p:Paper)

            // Optional MeSH filter
            OPTIONAL MATCH (p)-[:HAS_MESH_TERM]->(m:MeshTerm)
            WHERE $mesh_filter IS NULL OR toLower(m.term) CONTAINS toLower($mesh_filter)

            MATCH (p)<-[:MENTIONED_IN]-(g2:Gene)
            WHERE g2 <> g
            RETURN g2.name AS gene,
                COUNT(DISTINCT p) AS shared_papers,
                COLLECT(DISTINCT p.pmid)[..5] AS example_pmids
            ORDER BY shared_papers DESC
            LIMIT 10
        """
        return self.query(cypher, {"target_gene": target_gene, "mesh_filter": mesh_filter})

    def get_entity_community(
        self, entity_name: str,
    ) -> list[dict[str, Any]]:
        """Get other entities in the same community as the target entity.

        Requires community_id property from Louvain (run graph analytics first).
        """
        cypher = """
            MATCH (e1:ExtractedEntity)
            WHERE toLower(e1.name) = toLower($entity_name)
              AND e1.community_id IS NOT NULL
            WITH e1.community_id AS cid
            MATCH (e2:ExtractedEntity {community_id: cid})
            WHERE toLower(e2.name) <> toLower($entity_name)
              AND e2.confidence >= 0.75
              AND size(e2.name) <= 60
            RETURN e2.name AS entity, e2.type AS type,
                   e2.pagerank AS pagerank,
                   cid AS community_id
            ORDER BY e2.pagerank DESC
            LIMIT 15
        """
        return self.query(cypher, {"entity_name": entity_name})

    def get_top_central_entities(
        self, entity_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get the most central entities by PageRank.

        Requires pagerank property (run graph analytics first).
        """
        type_clause = "AND e.type = $entity_type" if entity_type else ""
        cypher = f"""
            MATCH (e:ExtractedEntity)
            WHERE e.pagerank IS NOT NULL
              AND e.confidence >= 0.75
              AND size(e.name) <= 60
              {type_clause}
            RETURN e.name AS entity, e.type AS type,
                   e.pagerank AS pagerank,
                   e.betweenness AS betweenness,
                   e.community_id AS community_id
            ORDER BY e.pagerank DESC
            LIMIT 15
        """
        params: dict[str, Any] = {}
        if entity_type:
            params["entity_type"] = entity_type
        return self.query(cypher, params)

    def get_research_opportunities(
        self, entity_name: str,
    ) -> list[dict[str, Any]]:
        """Get predicted cross-type connections for an entity — research opportunities.

        Focused on Gene↔Protein and Gene↔Disease predictions — the most
        actionable connections for biomedical research. Returns entities that
        appear in similar research contexts but never co-occur in a paper.

        Requires PREDICTED_LINK relationships (run graph analytics first).
        """
        cypher = """
            // Exclude entities with too many predicted links (hubs / noise)
            MATCH (noisy)-[:PREDICTED_LINK]-()
            WITH noisy, count(*) AS lc
            WHERE lc > 50
            WITH collect(noisy.name) AS noisy_names
            MATCH (e1)-[r:PREDICTED_LINK]-(e2)
            WHERE toLower(e1.name) = toLower($entity_name)
              AND NOT e1.name IN noisy_names AND NOT e2.name IN noisy_names
              AND (
                (e1.type = 'Gene' AND e2.type IN ['Protein', 'Disease'])
                OR (e1.type = 'Protein' AND e2.type IN ['Gene', 'Disease'])
                OR (e1.type = 'Disease' AND e2.type IN ['Gene', 'Protein'])
              )
              AND e2.confidence >= 0.90
              AND CASE
                WHEN e2.type IN ['Gene', 'Protein']
                  THEN NOT e2.name CONTAINS ' ' AND size(e2.name) <= 20
                ELSE size(e2.name) <= 40
                  AND NOT toLower(e2.name) ENDS WITH 'diseases'
                  AND NOT toLower(e2.name) ENDS WITH 'disorders'
              END
            RETURN e2.name AS entity, e2.type AS type,
                   r.similarity AS similarity
            ORDER BY r.similarity DESC
            LIMIT 10
        """
        return self.query(cypher, {"entity_name": entity_name})
