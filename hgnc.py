"""HGNC gene name and alias lookup module."""

import csv
from pathlib import Path


class HGNCLookup:
    """Lookup gene symbols and protein name aliases from HGNC data."""

    def __init__(self, hgnc_file: str = "hgnc_2026.txt"):
        """Initialize HGNC lookup from tab-delimited file."""
        self.hgnc_file = Path(hgnc_file)
        self._symbol_to_aliases = {}  # approved symbol -> set of all aliases
        self._alias_to_symbol = {}    # any alias -> approved symbol
        self._symbol_to_data = {}     # approved symbol -> raw HGNC data dict
        self._load_data()

    def _load_data(self):
        """Load HGNC data into lookup dictionaries."""
        if not self.hgnc_file.exists():
            raise FileNotFoundError(f"HGNC file not found: {self.hgnc_file}")

        with open(self.hgnc_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                # Skip withdrawn/removed entries
                status = row.get('Status', '')
                if 'Withdrawn' in status or 'Entry Withdrawn' in status:
                    continue

                approved_symbol = row.get('Approved symbol', '').strip()
                if not approved_symbol:
                    continue

                # Collect all aliases (gene symbols only, not protein names per requirement)
                aliases = set()
                aliases.add(approved_symbol)  # Include the approved symbol itself

                # Previous symbols
                prev = row.get('Previous symbols', '').strip()
                if prev:
                    aliases.update(s.strip() for s in prev.split(',') if s.strip())

                # Alias symbols
                alias_syms = row.get('Alias symbols', '').strip()
                if alias_syms:
                    aliases.update(s.strip() for s in alias_syms.split(',') if s.strip())

                # Alias names (protein names) - include these as well
                alias_names = row.get('Alias names', '').strip()
                if alias_names:
                    # Parse quoted comma-separated list
                    names = []
                    if alias_names.startswith('"') and alias_names.endswith('"'):
                        alias_names = alias_names[1:-1]
                    for name in alias_names.split('", "'):
                        name = name.replace('"', '').strip()
                        if name:
                            names.append(name)
                    aliases.update(names)

                # Store mappings
                self._symbol_to_aliases[approved_symbol] = aliases
                self._symbol_to_data[approved_symbol] = row
                for alias in aliases:
                    self._alias_to_symbol[alias.lower()] = approved_symbol

    def get_canonical_symbol(self, query: str) -> str | None:
        """Get the approved HGNC symbol for a query (gene name or alias).

        Returns the canonical symbol if found, otherwise None.
        """
        query_lower = query.strip().lower()
        return self._alias_to_symbol.get(query_lower)

    def get_all_aliases(self, gene: str) -> set[str]:
        """Get all aliases for a gene (including the approved symbol).

        Args:
            gene: Gene name or alias

        Returns:
            Set of all known aliases, or empty set if not found
        """
        canonical = self.get_canonical_symbol(gene)
        if canonical:
            return self._symbol_to_aliases.get(canonical, set())
        return set()

    def expand_query(self, gene: str) -> list[str]:
        """Expand a gene query to include the canonical symbol and top aliases.

        Returns a list of terms to search, with canonical symbol first.
        Limits to gene/protein symbols only (excludes long protein names).
        """
        canonical = self.get_canonical_symbol(gene)
        if not canonical:
            # Not found in HGNC, return original query
            return [gene.strip()]

        aliases = self.get_all_aliases(gene)

        # Filter to short symbols (likely gene symbols) and the canonical name
        # Exclude long descriptive names (> 20 chars is likely a full protein name)
        short_aliases = {a for a in aliases if len(a) <= 20}

        # Put canonical first, then other short aliases
        result = [canonical]
        for alias in sorted(short_aliases):
            if alias != canonical and alias not in result:
                result.append(alias)

        return result

    def get_alias_info(self, gene: str) -> dict | None:
        """Get structured alias information for display.

        Returns a dict with:
        - approved_symbol: Official HGNC symbol
        - approved_name: Full gene name
        - previous_symbols: List of previous gene symbols
        - alias_symbols: List of alternate gene symbols
        - alias_names: List of protein names

        Returns None if gene not found.
        """
        canonical = self.get_canonical_symbol(gene)
        if not canonical:
            return None

        data = self._symbol_to_data.get(canonical)
        if not data:
            return None

        def parse_list(value: str) -> list[str]:
            """Parse comma-separated or quoted list."""
            if not value.strip():
                return []

            # Handle quoted names like "name1", "name2"
            if '"' in value:
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                items = [item.replace('"', '').strip() for item in value.split('", "')]
                return [item for item in items if item]

            # Handle simple comma-separated
            return [item.strip() for item in value.split(',') if item.strip()]

        return {
            'approved_symbol': canonical,
            'approved_name': data.get('Approved name', '').strip(),
            'previous_symbols': parse_list(data.get('Previous symbols', '')),
            'alias_symbols': parse_list(data.get('Alias symbols', '')),
            'alias_names': parse_list(data.get('Alias names', '')),
        }
