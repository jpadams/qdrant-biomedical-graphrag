"""Graph Data Science analytics via Neo4j Aura Graph Analytics.

Creates an ephemeral GDS session, projects the entity co-occurrence graph,
runs community detection, centrality, and link prediction, then writes
results back to AuraDB.
"""

from __future__ import annotations

from datetime import timedelta

from biomedical_graphrag.config import settings
from biomedical_graphrag.utils.logger_util import setup_logging

logger = setup_logging()

# Co-occurrence projection: entities connected through shared papers.
# Filters: high confidence, name quality, and minimum 2 shared papers
# to keep the projected graph within memory limits.
# Uses gds.graph.project.remote() for Aura Graph Analytics compatibility.
ENTITY_COOCCURRENCE_PROJECTION = """
    MATCH (e1:ExtractedEntity)<-[:MENTIONED_IN_ABSTRACT]-(p:Paper)
          -[:MENTIONED_IN_ABSTRACT]->(e2:ExtractedEntity)
    WHERE e1 <> e2
      AND e1.confidence >= 0.85
      AND e2.confidence >= 0.85
      AND size(e1.name) <= 40
      AND size(e2.name) <= 40
      AND id(e1) < id(e2)
    WITH e1, e2, count(DISTINCT p) AS weight
    WHERE weight >= 2
    RETURN gds.graph.project.remote(e1, e2, {
      sourceNodeLabels: labels(e1),
      targetNodeLabels: labels(e2),
      relationshipType: 'CO_OCCURS_WITH',
      relationshipProperties: {weight: weight}
    })
"""


class GraphAnalytics:
    """Run GDS algorithms via Aura Graph Analytics sessions."""

    def __init__(self) -> None:
        self._gds = None
        self._graph = None

    def create_session(self) -> None:
        """Create an Aura Graph Analytics session."""
        from graphdatascience.session import (
            AuraAPICredentials,
            DbmsConnectionInfo,
            GdsSessions,
            SessionMemory,
        )

        cfg = settings.gds
        if not cfg.client_id or not cfg.client_secret.get_secret_value():
            raise ValueError(
                "GDS credentials not configured. Set GDS__CLIENT_ID and GDS__CLIENT_SECRET."
            )

        memory_map = {
            "2GB": SessionMemory.m_2GB,
            "4GB": SessionMemory.m_4GB,
            "8GB": SessionMemory.m_8GB,
        }
        session_memory = memory_map.get(cfg.session_memory, SessionMemory.m_2GB)

        creds_kwargs: dict = {
            "client_id": cfg.client_id,
            "client_secret": cfg.client_secret.get_secret_value(),
        }
        if cfg.project_id:
            creds_kwargs["project_id"] = cfg.project_id

        api_credentials = AuraAPICredentials(**creds_kwargs)
        sessions = GdsSessions(api_credentials=api_credentials)

        neo4j_cfg = settings.neo4j
        db_connection = DbmsConnectionInfo(
            username=neo4j_cfg.username,
            password=neo4j_cfg.password.get_secret_value(),
            aura_instance_id=cfg.aura_instance_id,
        )

        logger.info(f"Creating GDS session (memory={cfg.session_memory}, ttl={cfg.session_ttl_minutes}m)...")
        self._gds = sessions.get_or_create(
            session_name="biomedical-graphrag-analytics",
            memory=session_memory,
            db_connection=db_connection,
            ttl=timedelta(minutes=cfg.session_ttl_minutes),
        )
        logger.info("✅ GDS session created.")

    def project_entity_graph(self) -> None:
        """Project entity co-occurrence graph into GDS."""
        if self._gds is None:
            raise RuntimeError("No GDS session. Call create_session() first.")

        # Drop existing projection if it exists
        try:
            existing = self._gds.graph.get("entity-cooccurrence")
            existing.drop()
            logger.info("Dropped existing entity-cooccurrence projection.")
        except Exception:
            pass

        logger.info("Projecting entity co-occurrence graph...")
        self._graph, result = self._gds.graph.project(
            "entity-cooccurrence",
            ENTITY_COOCCURRENCE_PROJECTION,
        )
        node_count = result.get("nodeCount", "?")
        rel_count = result.get("relationshipCount", "?")
        logger.info(f"✅ Projected graph: {node_count} nodes, {rel_count} relationships.")

    def run_community_detection(self) -> dict:
        """Run Louvain community detection and write community_id back to DB."""
        if self._graph is None:
            raise RuntimeError("No projected graph. Call project_entity_graph() first.")

        logger.info("Running Louvain community detection...")
        result = self._gds.louvain.write(
            self._graph,
            writeProperty="community_id",
            relationshipWeightProperty="weight",
        )
        communities = result.get("communityCount", "?")
        modularity = result.get("modularity", "?")
        logger.info(f"✅ Louvain: {communities} communities, modularity={modularity}")
        return {"community_count": communities, "modularity": modularity}

    def run_centrality(self) -> dict:
        """Run PageRank and Betweenness centrality, write scores back to DB."""
        if self._graph is None:
            raise RuntimeError("No projected graph. Call project_entity_graph() first.")

        logger.info("Running PageRank...")
        pr_result = self._gds.pageRank.write(
            self._graph,
            writeProperty="pagerank",
            relationshipWeightProperty="weight",
        )
        logger.info(f"✅ PageRank complete. Iterations: {pr_result.get('ranIterations', '?')}")

        logger.info("Running Betweenness Centrality...")
        bc_result = self._gds.betweenness.write(
            self._graph,
            writeProperty="betweenness",
        )
        logger.info(f"✅ Betweenness complete.")

        return {
            "pagerank_iterations": pr_result.get("ranIterations"),
            "betweenness_nodes": bc_result.get("nodePropertiesWritten"),
        }

    def run_link_prediction(self, top_k: int = 50) -> list[dict]:
        """Run link prediction on the GDS projected graph.

        Uses node similarity on the in-memory graph to find entity pairs
        that are structurally similar (many shared neighbors) but not
        directly connected. These are predicted missing links.
        """
        if self._gds is None or self._graph is None:
            raise RuntimeError("No projected graph. Call project_entity_graph() first.")

        logger.info("Running link prediction (Node Similarity on GDS)...")

        # Use nodeSimilarity to find pairs with high Jaccard similarity
        # that share many neighbors in the co-occurrence graph.
        # mutate into the graph, then write to DB as PREDICTED_LINK relationships.
        self._gds.nodeSimilarity.mutate(
            self._graph,
            similarityCutoff=0.1,
            topK=5,
            mutateRelationshipType="PREDICTED_LINK",
            mutateProperty="similarity",
        )

        # Write the predicted links back to AuraDB
        self._gds.graph.relationship.write(
            self._graph,
            relationship_type="PREDICTED_LINK",
            relationship_property="similarity",
        )

        # Now query the written relationships from AuraDB
        from neo4j import GraphDatabase

        neo4j_cfg = settings.neo4j
        driver = GraphDatabase.driver(
            neo4j_cfg.uri,
            auth=(neo4j_cfg.username, neo4j_cfg.password.get_secret_value()),
        )

        try:
            with driver.session() as session:
                result = session.run(
                    """
                    MATCH (e1:ExtractedEntity)-[r:PREDICTED_LINK]->(e2:ExtractedEntity)
                    WHERE e1.type <> e2.type
                    RETURN e1.name AS entity1, e1.type AS type1,
                           e2.name AS entity2, e2.type AS type2,
                           r.similarity AS similarity
                    ORDER BY r.similarity DESC
                    LIMIT $top_k
                    """,
                    {"top_k": top_k},
                )
                predictions = [dict(record) for record in result]

                # Also count total
                count_result = session.run(
                    "MATCH ()-[r:PREDICTED_LINK]->() RETURN count(r) AS total"
                )
                total = count_result.single()["total"]
        finally:
            driver.close()

        logger.info(
            f"✅ Link prediction: {total} total PREDICTED_LINK relationships written, "
            f"{len(predictions)} cross-type predictions returned."
        )
        return predictions

    def close(self) -> None:
        """Drop the projected graph (session auto-expires via TTL)."""
        if self._graph is not None:
            try:
                self._graph.drop()
                logger.info("Dropped projected graph.")
            except Exception:
                pass
        self._graph = None
        self._gds = None

    def run_all(self) -> dict:
        """Run the full analytics pipeline."""
        self.create_session()
        try:
            self.project_entity_graph()
            community_stats = self.run_community_detection()
            centrality_stats = self.run_centrality()
            predictions = self.run_link_prediction()
            return {
                "communities": community_stats,
                "centrality": centrality_stats,
                "predictions_count": len(predictions),
                "top_predictions": predictions[:10],
            }
        finally:
            self.close()


def main() -> None:
    """CLI entry point: run full graph analytics pipeline."""
    analytics = GraphAnalytics()
    stats = analytics.run_all()

    print("\n=== Graph Analytics Results ===")
    print(f"Communities: {stats['communities']}")
    print(f"Centrality: {stats['centrality']}")
    print(f"\nTop predicted connections ({stats['predictions_count']} total):")
    for p in stats["top_predictions"]:
        print(
            f"  {p['entity1']} ({p['type1']}) <-> {p['entity2']} ({p['type2']}) "
            f"| similarity: {p['similarity']}"
        )


if __name__ == "__main__":
    main()
