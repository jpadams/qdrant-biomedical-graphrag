"""Gene gazetteer extractor — dictionary matching against known gene names/aliases."""

from __future__ import annotations

import re

from biomedical_graphrag.extraction.base import ExtractedEntity, ExtractionResult
from biomedical_graphrag.utils.json_util import load_gene_json
from biomedical_graphrag.utils.logger_util import setup_logging

logger = setup_logging()


class GeneGazetteerExtractor:
    """Scan text for known gene names and aliases from the gene dataset.

    Builds a compiled regex from all gene names and aliases at init time,
    then runs fast dictionary matching against each abstract.
    """

    def __init__(
        self,
        min_name_length: int = 3,
        name_confidence: float = 0.95,
        alias_confidence: float = 0.85,
        context_window: int = 50,
    ) -> None:
        self.min_name_length = min_name_length
        self.name_confidence = name_confidence
        self.alias_confidence = alias_confidence
        self.context_window = context_window

        self._gene_lookup: dict[str, dict] = {}
        self._pattern: re.Pattern | None = None
        self._build_index()

    def _build_index(self) -> None:
        """Load gene data and compile the matching regex."""
        data = load_gene_json()
        genes = data.get("genes", [])

        # Map each name/alias (lowercased) to gene metadata
        # _canonical maps lowercase -> preferred display form
        self._canonical: dict[str, str] = {}
        terms: list[str] = []

        for gene in genes:
            gene_id = gene.get("gene_id", "")
            name = gene.get("name", "").strip()
            organism = gene.get("organism", "")
            aliases_raw = gene.get("aliases", "")

            if not name:
                continue

            # Primary name
            if len(name) >= self.min_name_length:
                key = name.lower()
                if key not in self._gene_lookup:
                    self._gene_lookup[key] = {
                        "gene_id": gene_id,
                        "canonical_name": name,
                        "organism": organism,
                        "match_type": "name",
                    }
                    self._canonical[key] = name
                    terms.append(name)

            # Aliases
            if aliases_raw:
                for alias in aliases_raw.split(", "):
                    alias = alias.strip()
                    if not alias or len(alias) < self.min_name_length:
                        continue
                    key = alias.lower()
                    if key not in self._gene_lookup:
                        self._gene_lookup[key] = {
                            "gene_id": gene_id,
                            "canonical_name": name,
                            "organism": organism,
                            "match_type": "alias",
                        }
                        self._canonical[key] = alias
                        terms.append(alias)

        if not terms:
            logger.warning("Gene gazetteer: no terms to index")
            return

        # Sort by length descending so longer matches take priority
        terms.sort(key=len, reverse=True)
        escaped = [re.escape(t) for t in terms]
        self._pattern = re.compile(
            r"\b(?:" + "|".join(escaped) + r")\b",
            re.IGNORECASE,
        )
        logger.info(f"Gene gazetteer: indexed {len(terms)} terms from {len(genes)} genes")

    async def extract(self, text: str) -> ExtractionResult:
        """Extract gene entities by dictionary matching."""
        if self._pattern is None:
            return ExtractionResult(source_text=text)

        entities: list[ExtractedEntity] = []
        seen_positions: set[tuple[int, int]] = set()

        for match in self._pattern.finditer(text):
            start, end = match.start(), match.end()
            if (start, end) in seen_positions:
                continue
            seen_positions.add((start, end))

            matched_text = match.group()
            key = matched_text.lower()
            info = self._gene_lookup.get(key)
            if info is None:
                continue

            confidence = (
                self.name_confidence
                if info["match_type"] == "name"
                else self.alias_confidence
            )

            ctx_start = max(0, start - self.context_window)
            ctx_end = min(len(text), end + self.context_window)

            entities.append(
                ExtractedEntity(
                    name=info["canonical_name"],
                    type="Gene",
                    confidence=confidence,
                    start_pos=start,
                    end_pos=end,
                    context=text[ctx_start:ctx_end],
                    extractor="gene_gazetteer",
                    attributes={
                        "gene_id": info["gene_id"],
                        "organism": info["organism"],
                        "match_type": info["match_type"],
                    },
                )
            )

        return ExtractionResult(entities=entities, source_text=text)
