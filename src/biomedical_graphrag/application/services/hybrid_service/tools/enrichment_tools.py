"""
Enrichment tools definition for Neo4j graph queries.
"""

# ----------------------------
# Enrichment tools definition
# ----------------------------
NEO4J_ENRICHMENT_TOOLS = [
    {
        "type": "function",
        "name": "get_collaborators_with_topics",
        "description": "Get collaborators for an author filtered by MeSH topics.",
        "parameters": {
            "type": "object",
            "properties": {
                "author_name": {"type": "string"},
                "topics": {"type": "array", "items": {"type": "string"}},
                "require_all": {"type": "boolean"},
            },
            "required": ["author_name", "topics"],
        },
    },
    {
        "type": "function",
        "name": "get_related_papers_by_mesh",
        "description": "Get papers related by MeSH terms to a given PMID.",
        "parameters": {
            "type": "object",
            "properties": {"pmid": {"type": "string"}},
            "required": ["pmid"],
        },
    },
    {
        "type": "function",
        "name": "get_entities_for_papers",
        "description": (
            "Get all biomedical entities (genes, diseases, drugs, proteins, etc.) "
            "extracted from specific papers by PMID. Returns entities grounded in "
            "the actual paper abstracts, grouped by type. Use this to understand "
            "what entities are mentioned in the retrieved papers."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pmids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of PMIDs to query for extracted entities.",
                },
                "entity_type_filter": {
                    "type": "string",
                    "description": (
                        "Optional: filter by entity type. "
                        "One of: Gene, Protein, Disease, Drug, CellType, Organism, "
                        "Technique, BiologicalProcess, AnatomicalStructure."
                    ),
                },
            },
            "required": ["pmids"],
        },
    },
    {
        "type": "function",
        "name": "get_entity_cooccurrence",
        "description": (
            "Find entities (genes, diseases, drugs, proteins, techniques, etc.) "
            "co-mentioned in the same papers as a target entity. Uses NER-extracted "
            "entities from paper abstracts. Reveals biological associations based on "
            "co-mention frequency across the literature."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "entity_name": {
                    "type": "string",
                    "description": (
                        "Name of the target entity (e.g., 'CREBBP', 'glioblastoma', "
                        "'doxorubicin'). Case-insensitive, supports partial match."
                    ),
                },
                "synonyms": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional synonyms or abbreviations for the entity to broaden "
                        "the search (e.g., ['DLBCL', 'diffuse large B-cell lymphoma'] "
                        "for 'non-Hodgkin lymphoma'). Use your biomedical knowledge to "
                        "provide common alternative names."
                    ),
                },
                "entity_type_filter": {
                    "type": "string",
                    "description": (
                        "Optional: filter co-occurring entities by type. "
                        "One of: Gene, Protein, Disease, Drug, CellType, Organism, "
                        "Technique, BiologicalProcess, AnatomicalStructure."
                    ),
                },
            },
            "required": ["entity_name"],
        },
    },
    {
        "type": "function",
        "name": "get_genes_in_same_papers",
        "description": (
            "Find genes that co-occur in the same papers as a specified target gene, "
            "optionally filtered by a MeSH topic (e.g., 'cancer', 'HIV'). "
            "This reveals potential biological associations based on co-mention frequency."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "target_gene": {
                    "type": "string",
                    "description": (
                        "Name or alias of the target gene (e.g., 'TP53', 'CCR5', 'gag'). "
                        "Search is case-insensitive and supports partial matches."
                    ),
                },
                "mesh_filter": {
                    "type": "string",
                    "description": (
                        "Optional MeSH term substring to filter relevant papers "
                        "(e.g., 'cancer', 'immunity', 'HIV')."
                    ),
                },
            },
            "required": ["target_gene"],
        },
    },
]
