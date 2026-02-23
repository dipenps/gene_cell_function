"""PubMed search module for gene-cell function queries."""

from dataclasses import dataclass
from Bio import Entrez, Medline
import xml.etree.ElementTree as ET


class PubMedError(Exception):
    """Raised when PubMed search or fetch fails."""


@dataclass
class Article:
    pmid: str
    title: str
    authors: list[str]
    journal: str
    year: str
    abstract: str
    doi: str


def search_pubmed(
    gene: str,
    cell_type: str,
    max_results: int = 20,
    email: str = "user@example.com",
    use_contextual_query: bool = True,
) -> list[Article]:
    """Search PubMed for articles about a gene's function in a cell type.

    Args:
        gene: Gene symbol to search
        cell_type: Cell type to search
        max_results: Maximum number of results
        email: NCBI requires an email for API access
        use_contextual_query: If True, use improved query with functional keywords

    Returns:
        A list of Article dataclasses sorted by relevance.
    """
    Entrez.email = email

    if use_contextual_query:
        # Improved query: gene + cell type + functional keywords
        # This reduces false positives by requiring functional context terms
        query = (
            f'"{gene}"[Title/Abstract] AND "{cell_type}"[Title/Abstract] AND ('
            f'"expression"[Title/Abstract] OR '
            f'"function"[Title/Abstract] OR '
            f'"role"[Title/Abstract] OR '
            f'"activity"[Title/Abstract] OR '
            f'"regulation"[Title/Abstract] OR '
            f'"signaling"[Title/Abstract] OR '
            f'"pathway"[Title/Abstract] OR '
            f'"differentiation"[Title/Abstract] OR '
            f'"proliferation"[Title/Abstract] OR '
            f'"activation"[Title/Abstract]'
            f')'
        )
    else:
        # Original simple co-occurrence query
        query = f'"{gene}"[Title/Abstract] AND "{cell_type}"[Title/Abstract]'

    # Search for matching PMIDs
    try:
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, sort="relevance")
        search_results = Entrez.read(handle)
        handle.close()
    except Exception as e:
        raise PubMedError(f"PubMed search failed: {e}") from e

    id_list = search_results.get("IdList", [])
    if not id_list:
        return []

    # Fetch article details
    try:
        handle = Entrez.efetch(db="pubmed", id=id_list, rettype="xml", retmode="xml")
        xml_data = handle.read()
        handle.close()
    except Exception as e:
        raise PubMedError(f"PubMed fetch failed: {e}") from e

    return _parse_articles(xml_data)


def _parse_articles(xml_data: bytes) -> list[Article]:
    """Parse PubMed XML response into Article dataclasses."""
    root = ET.fromstring(xml_data)
    articles = []

    for article_elem in root.findall(".//PubmedArticle"):
        medline = article_elem.find("MedlineCitation")
        if medline is None:
            continue

        pmid_elem = medline.find("PMID")
        pmid = pmid_elem.text if pmid_elem is not None else ""

        art = medline.find("Article")
        if art is None:
            continue

        # Title
        title_elem = art.find("ArticleTitle")
        title = _get_text(title_elem) if title_elem is not None else ""

        # Authors
        authors = []
        author_list = art.find("AuthorList")
        if author_list is not None:
            for author in author_list.findall("Author"):
                last = author.find("LastName")
                fore = author.find("ForeName")
                if last is not None and fore is not None:
                    authors.append(f"{fore.text} {last.text}")
                elif last is not None:
                    authors.append(last.text)

        # Journal
        journal_elem = art.find("Journal/Title")
        journal = journal_elem.text if journal_elem is not None else ""

        # Year
        year = ""
        pub_date = art.find("Journal/JournalIssue/PubDate")
        if pub_date is not None:
            year_elem = pub_date.find("Year")
            if year_elem is not None:
                year = year_elem.text
            else:
                medline_date = pub_date.find("MedlineDate")
                if medline_date is not None and medline_date.text:
                    year = medline_date.text[:4]

        # Abstract — handle structured abstracts with labeled sections
        abstract = ""
        abstract_elem = art.find("Abstract")
        if abstract_elem is not None:
            parts = []
            for text_elem in abstract_elem.findall("AbstractText"):
                label = text_elem.get("Label", "")
                text = _get_text(text_elem)
                if label:
                    parts.append(f"{label}: {text}")
                else:
                    parts.append(text)
            abstract = " ".join(parts)

        # DOI
        doi = ""
        article_data = article_elem.find("PubmedData")
        if article_data is not None:
            for id_elem in article_data.findall("ArticleIdList/ArticleId"):
                if id_elem.get("IdType") == "doi":
                    doi = id_elem.text or ""
                    break

        articles.append(Article(
            pmid=pmid,
            title=title,
            authors=authors,
            journal=journal,
            year=year,
            abstract=abstract,
            doi=doi,
        ))

    return articles


def _get_text(elem) -> str:
    """Extract all text content from an XML element, including mixed content."""
    return "".join(elem.itertext()).strip()
