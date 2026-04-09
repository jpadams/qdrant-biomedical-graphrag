"""Biomedical entity type definitions and label mappings for extractors."""

BIOMEDICAL_ENTITY_TYPES = [
    "Gene",
    "Protein",
    "Disease",
    "Drug",
    "CellType",
    "Organism",
    "Technique",
    "BiologicalProcess",
    "AnatomicalStructure",
]

BIOMEDICAL_RELATION_TYPES = [
    "TARGETS",
    "ASSOCIATED_WITH",
    "TREATS",
    "EXPRESSED_IN",
    "INHIBITS",
    "ACTIVATES",
    "DERIVED_FROM",
    "INTERACTS_WITH",
]

# ── GLiREL relation label descriptions (improve zero-shot accuracy) ───
GLIREL_RELATION_LABELS: dict[str, str] = {
    "TARGETS": "targets, acts on, or binds to (e.g., a drug targets a protein, a technique targets a gene)",
    "ASSOCIATED_WITH": "is associated with or related to (e.g., a gene is associated with a disease)",
    "TREATS": "is used to treat or manage (e.g., a drug treats a disease)",
    "EXPRESSED_IN": "is expressed in or found in (e.g., a gene is expressed in a cell type or tissue)",
    "INHIBITS": "inhibits, suppresses, or blocks (e.g., a drug inhibits an enzyme or process)",
    "ACTIVATES": "activates, induces, or promotes (e.g., a protein activates a signaling pathway)",
    "DERIVED_FROM": "is derived from or originates from (e.g., a cell line is derived from a tissue)",
    "INTERACTS_WITH": "interacts with or binds to (e.g., two proteins form a complex)",
}

# ── GLiNER label descriptions (improve zero-shot accuracy) ─────────────
GLINER_BIOMEDICAL_LABELS: dict[str, str] = {
    "Gene": "A named gene or gene symbol (e.g., TP53, BRCA1, NOTCH1, RB1, CCND1)",
    "Protein": (
        "A named protein, enzyme, receptor, or protein complex "
        "(e.g., cyclin D1, p53, IL-15, CD70, Cas9, CAR)"
    ),
    "Disease": (
        "A disease, disorder, syndrome, or medical condition "
        "(e.g., small cell lung cancer, glioblastoma, endometriosis, dermatophytosis)"
    ),
    "Drug": (
        "A drug, pharmaceutical compound, chemical agent, or therapeutic substance "
        "(e.g., terbinafine, doxorubicin, antifungals)"
    ),
    "CellType": (
        "A cell type, cell line, or cell population "
        "(e.g., CAR-T cells, NK cells, iPSC, HEK293, pluripotent stem cells)"
    ),
    "Organism": (
        "A species, organism, or taxonomic group "
        "(e.g., Homo sapiens, mouse, E. coli, salmon, Drosophila)"
    ),
    "Technique": (
        "A laboratory technique, experimental method, or technology "
        "(e.g., CRISPR-Cas9, ZFN, TALEN, whole-genome sequencing, flow cytometry)"
    ),
    "BiologicalProcess": (
        "A biological process, pathway, or molecular mechanism "
        "(e.g., Notch signaling, chromothripsis, apoptosis, gene editing, epitranscriptome)"
    ),
    "AnatomicalStructure": (
        "An organ, tissue, body part, or anatomical structure "
        "(e.g., lung, ovary, endometrium, liver, blood, brain)"
    ),
}

# ── LLM extraction prompt ──────────────────────────────────────────────
LLM_EXTRACTION_SYSTEM_PROMPT = """\
You are a biomedical named entity recognition and relationship extraction system.
Given a PubMed abstract, extract all biomedical entities and relationships.

## Entity Types
- Gene: Named genes or gene symbols (e.g., TP53, BRCA1, NOTCH1)
- Protein: Named proteins, enzymes, receptors (e.g., cyclin D1, p53, IL-15, CD70)
- Disease: Diseases, disorders, conditions (e.g., SCLC, glioblastoma, endometriosis)
- Drug: Drugs, chemicals, therapeutic substances (e.g., terbinafine, doxorubicin)
- CellType: Cell types, cell lines (e.g., CAR-T, NK cells, iPSC, HEK293)
- Organism: Species or organisms (e.g., Homo sapiens, mouse, E. coli)
- Technique: Lab techniques, methods (e.g., CRISPR-Cas9, ZFN, TALEN, WGS)
- BiologicalProcess: Biological processes, pathways (e.g., Notch signaling, apoptosis)
- AnatomicalStructure: Organs, tissues, body parts (e.g., lung, ovary, brain)

## Relationship Types
- TARGETS: A technique or drug targets a gene/protein/disease
- ASSOCIATED_WITH: An entity is associated with another (gene-disease, protein-process)
- TREATS: A drug or technique treats a disease
- EXPRESSED_IN: A gene/protein is expressed in a cell type or tissue
- INHIBITS: An entity inhibits a process or another entity
- ACTIVATES: An entity activates a process or another entity
- DERIVED_FROM: A cell type is derived from an organism or tissue
- INTERACTS_WITH: Two proteins/genes interact with each other

## Rules
- Extract SPECIFIC named entities, not generic terms like "gene" or "protein"
- Prefer the most specific name used in the text
- Assign confidence 0.0-1.0 based on how clearly the entity is mentioned
- For relationships, both source and target must be entities you extracted
- Do NOT extract MeSH-style broad categories; extract specific named things
"""

LLM_EXTRACTION_USER_PROMPT = """\
Extract all biomedical entities and relationships from this abstract.

Abstract:
{abstract}

Respond with JSON matching this exact schema:
{{
  "entities": [
    {{"name": "entity name", "type": "EntityType", "confidence": 0.9}}
  ],
  "relations": [
    {{"source": "entity1 name", "target": "entity2 name",
      "relation_type": "RELATION_TYPE", "confidence": 0.8}}
  ]
}}
"""
