"""Streamlit app for gene-cell function summarization."""

import datetime
import ollama
import pandas as pd
import streamlit as st
from pubmed import search_pubmed, PubMedError
from summarizer import summarize_gene_function, summarize_gene_function_brief, SummarizerError, LENGTH_PRESETS
from hgnc import HGNCLookup
from context_filter import RelevanceResult

st.set_page_config(page_title="Gene-Cell Function Summarizer", layout="wide")


@st.cache_resource(ttl=None, hash_funcs={type: id})
def load_hgnc(version="v3"):
    try:
        return HGNCLookup()
    except FileNotFoundError:
        return None


@st.cache_data(ttl=60)
def get_ollama_models() -> list[str]:
    """Detect available Ollama models. Refreshes every 60 seconds."""
    try:
        response = ollama.list()
        return [m["model"] for m in response.get("models", [])]
    except Exception:
        return []


def parse_gene_list(raw: str) -> list[str]:
    """Parse comma/newline/semicolon-separated gene names into a deduplicated list."""
    genes = []
    for part in raw.replace(";", ",").replace("\n", ",").split(","):
        g = part.strip()
        if g and g not in genes:
            genes.append(g)
    return genes


def resolve_gene(gene: str, hgnc) -> tuple[str, str | None, dict | None]:
    """Resolve a gene through HGNC. Returns (query, canonical_symbol, alias_info)."""
    if not hgnc:
        return gene, None, None
    canonical = hgnc.get_canonical_symbol(gene)
    alias_info = hgnc.get_alias_info(gene) if canonical else None
    return (canonical or gene), canonical, alias_info


def build_alias_table_md(alias_info: dict) -> str:
    """Build markdown alias table for a single gene."""
    md = "| Type | Names |\n|------|-------|\n"
    md += f"| **Approved Symbol** | {alias_info['approved_symbol']} |\n"
    md += f"| **Approved Name** | {alias_info['approved_name']} |\n"
    if alias_info['previous_symbols']:
        md += f"| **Previous Symbols** | {', '.join(alias_info['previous_symbols'])} |\n"
    if alias_info['alias_symbols']:
        md += f"| **Alias Symbols** | {', '.join(alias_info['alias_symbols'])} |\n"
    if alias_info['alias_names']:
        md += f"| **Protein Names** | {', '.join(alias_info['alias_names'])} |\n"
    return md


def build_alias_table_html(alias_info: dict) -> str:
    """Build HTML alias table for a single gene."""
    html = "    <table>\n        <tr><th>Type</th><th>Names</th></tr>\n"
    html += f"        <tr><td>Approved Symbol</td><td>{alias_info['approved_symbol']}</td></tr>\n"
    html += f"        <tr><td>Approved Name</td><td>{alias_info['approved_name']}</td></tr>\n"
    if alias_info['previous_symbols']:
        html += f"        <tr><td>Previous Symbols</td><td>{', '.join(alias_info['previous_symbols'])}</td></tr>\n"
    if alias_info['alias_symbols']:
        html += f"        <tr><td>Alias Symbols</td><td>{', '.join(alias_info['alias_symbols'])}</td></tr>\n"
    if alias_info['alias_names']:
        html += f"        <tr><td>Protein Names</td><td>{', '.join(alias_info['alias_names'])}</td></tr>\n"
    html += "    </table>\n"
    return html


def display_alias_table(alias_info: dict):
    """Display HGNC alias table in Streamlit."""
    table_data = [
        {"Type": "Approved Symbol", "Names": alias_info['approved_symbol']},
        {"Type": "Approved Name", "Names": alias_info['approved_name']},
    ]
    if alias_info['previous_symbols']:
        table_data.append({"Type": "Previous Symbols", "Names": ", ".join(alias_info['previous_symbols'])})
    if alias_info['alias_symbols']:
        table_data.append({"Type": "Alias Symbols", "Names": ", ".join(alias_info['alias_symbols'])})
    if alias_info['alias_names']:
        table_data.append({"Type": "Protein Names", "Names": ", ".join(alias_info['alias_names'])})
    st.dataframe(pd.DataFrame(table_data), hide_index=True, use_container_width=True)


def display_articles(articles: list):
    """Display source articles in expandable sections."""
    for article in articles:
        label = f"{article.title} ({article.year})"
        with st.expander(label):
            st.markdown(f"**Authors:** {', '.join(article.authors[:5])}{'...' if len(article.authors) > 5 else ''}")
            st.markdown(f"**Journal:** {article.journal}")
            st.markdown(f"**PMID:** [{article.pmid}](https://pubmed.ncbi.nlm.nih.gov/{article.pmid}/)")
            if article.doi:
                st.markdown(f"**DOI:** [{article.doi}](https://doi.org/{article.doi})")
            if article.abstract:
                st.markdown(f"**Abstract:** {article.abstract}")
            else:
                st.markdown("*No abstract available.*")


def display_filtered_results(filtered_results: list[RelevanceResult]):
    """Display excluded articles with reasoning."""
    for result in filtered_results:
        label = f"{result.article.title} ({result.article.year}) - {result.category}"
        with st.expander(label):
            st.markdown(f"**Relevance:** {result.category.replace('_', ' ').title()}")
            st.markdown(f"**Confidence:** {result.confidence:.2f}")
            st.markdown(f"**Reasoning:** {result.reasoning}")
            st.markdown(f"**PMID:** [{result.article.pmid}](https://pubmed.ncbi.nlm.nih.gov/{result.article.pmid}/)")
            if result.article.abstract:
                st.markdown(f"**Abstract:** {result.article.abstract}")


def articles_to_md(articles: list) -> str:
    """Convert articles list to markdown."""
    md = ""
    for i, a in enumerate(articles, 1):
        md += f"### {i}. {a.title} ({a.year})\n\n"
        md += f"**Authors:** {', '.join(a.authors)}\n\n"
        md += f"**Journal:** {a.journal}\n\n"
        md += f"**PMID:** [{a.pmid}](https://pubmed.ncbi.nlm.nih.gov/{a.pmid}/)\n\n"
        if a.doi:
            md += f"**DOI:** [{a.doi}](https://doi.org/{a.doi})\n\n"
        if a.abstract:
            md += f"**Abstract:** {a.abstract}\n\n"
        md += "---\n\n"
    return md


def articles_to_html(articles: list) -> str:
    """Convert articles list to collapsible HTML details elements."""
    html = ""
    for i, a in enumerate(articles, 1):
        html += f'    <details class="article">\n'
        html += f"        <summary>{i}. {a.title} ({a.year})</summary>\n"
        html += f"        <p><strong>Authors:</strong> {', '.join(a.authors)}</p>\n"
        html += f"        <p><strong>Journal:</strong> {a.journal}</p>\n"
        html += f'        <p><strong>PMID:</strong> <a href="https://pubmed.ncbi.nlm.nih.gov/{a.pmid}/">{a.pmid}</a></p>\n'
        if a.doi:
            html += f'        <p><strong>DOI:</strong> <a href="https://doi.org/{a.doi}">{a.doi}</a></p>\n'
        if a.abstract:
            html += f'        <div class="abstract"><strong>Abstract:</strong> {a.abstract}</div>\n'
        html += "    </details>\n"
    return html


HTML_HEADER = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 900px; margin: 40px auto; padding: 20px; line-height: 1.6; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; border-bottom: 2px solid #bdc3c7; padding-bottom: 5px; }}
        h3 {{ color: #7f8c8d; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #3498db; color: white; }}
        .meta {{ color: #7f8c8d; font-size: 0.9em; margin-bottom: 20px; }}
        details.article {{ margin: 10px 0; padding: 15px; background: #f8f9fa; border-left: 4px solid #3498db; }}
        details.article summary {{ cursor: pointer; font-weight: bold; color: #34495e; padding: 5px 0; }}
        details.article summary:hover {{ color: #3498db; }}
        details.article[open] summary {{ margin-bottom: 10px; border-bottom: 1px solid #ddd; padding-bottom: 10px; }}
        .abstract {{ margin-top: 10px; text-align: justify; }}
        .gene-section {{ margin: 30px 0; padding: 20px; border: 1px solid #e0e0e0; border-radius: 8px; }}
    </style>
</head>
<body>
"""

HTML_FOOTER = "\n</body>\n</html>"

# --- Sidebar ---
with st.sidebar:
    st.header("Settings")
    available_models = get_ollama_models()
    if available_models:
        model = st.selectbox("LLM Model", options=available_models)
    else:
        st.warning("No Ollama models detected. Is `ollama serve` running?")
        model = st.text_input("LLM Model (manual)", value="llama3.2:latest")
    max_articles = st.slider("Max articles", min_value=5, max_value=30, value=20)
    summary_length = st.select_slider(
        "Summary length",
        options=list(LENGTH_PRESETS.keys()),
        value="medium",
        help="Short: 1-2 paragraphs | Medium: 2-4 paragraphs | Long: 5-7 paragraphs",
    )
    email = st.text_input("NCBI Email", value="user@example.com", help="Required by NCBI for API access")
    reviews_only = st.toggle("Reviews only", value=True, help="Filter to review articles only for higher-quality summaries")

# --- Main area ---
st.title("Gene-Cell Function Summarizer")
st.markdown("Enter one or more gene names and a cell type to search PubMed and generate AI summaries.")

col1, col2 = st.columns(2)
with col1:
    gene_input = st.text_area("Gene Name(s)", placeholder="e.g. TP53\nor: TP53, FOXP3, CD3E", height=80)
with col2:
    cell_type = st.text_input("Cell Type", placeholder="e.g. T cells")

if st.button("Search & Summarize", type="primary"):
    genes = parse_gene_list(gene_input)
    if not genes:
        st.error("Please enter at least one gene name.")
    elif not cell_type.strip():
        st.error("Please enter a cell type.")
    else:
        hgnc = load_hgnc()
        multi_gene = len(genes) > 1
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')

        # Track all results for download report
        results = []  # list of dicts per gene

        if multi_gene:
            st.info(f"Processing **{len(genes)} genes**: {', '.join(genes)}")

        for gene in genes:
            gene_query, canonical_symbol, alias_info = resolve_gene(gene, hgnc)

            if multi_gene:
                st.divider()
                st.subheader(f"{canonical_symbol or gene}")

            if canonical_symbol and canonical_symbol != gene:
                st.caption(f"HGNC: {gene} -> {canonical_symbol}")

            # --- PubMed search ---
            with st.status(f"Searching PubMed for {gene_query}...", expanded=not multi_gene) as status:
                try:
                    articles = search_pubmed(
                        gene_query,
                        cell_type.strip(),
                        max_results=max_articles,
                        email=email,
                        use_contextual_query=True,
                        reviews_only=reviews_only,
                    )
                except PubMedError as e:
                    st.error(f"PubMed search failed for {gene_query}: {e}")
                    results.append({"gene": gene, "canonical": canonical_symbol, "alias_info": alias_info, "articles": [], "summary": None, "error": str(e)})
                    continue

                if not articles:
                    st.warning(
                        f'No articles found for **{gene_query}** in **{cell_type}**. '
                        "Try broader terms (e.g., shorter gene alias or general cell lineage)."
                    )
                    results.append({"gene": gene, "canonical": canonical_symbol, "alias_info": alias_info, "articles": [], "summary": None, "error": None})
                    continue

                search_msg = f"Found {len(articles)} articles for {gene_query}"
                if reviews_only:
                    search_msg += " [reviews only]"
                status.update(label=search_msg, state="complete")

            # --- LLM summarization ---
            summarize_fn = summarize_gene_function_brief if multi_gene else summarize_gene_function
            label = f"Summarizing {gene_query} ({summary_length})..."
            with st.status(label, expanded=not multi_gene) as status:
                try:
                    summary = summarize_fn(gene_query, cell_type.strip(), articles, model=model, summary_length=summary_length)
                except SummarizerError as e:
                    error_msg = str(e)
                    if "connection" in error_msg.lower() or "refused" in error_msg.lower():
                        st.error(
                            "Could not connect to Ollama. Make sure it is running:\n\n"
                            "```bash\nollama serve\n```"
                        )
                    else:
                        st.error(f"Summarization failed for {gene_query}: {e}")
                    results.append({"gene": gene, "canonical": canonical_symbol, "alias_info": alias_info, "articles": articles, "summary": None, "error": str(e)})
                    continue

                status.update(label=f"Summary complete for {gene_query}", state="complete")

            results.append({"gene": gene, "canonical": canonical_symbol, "alias_info": alias_info, "articles": articles, "summary": summary, "error": None})

            # --- Display summary ---
            st.markdown(summary)

            # --- Alias table ---
            if alias_info:
                with st.expander("Gene Aliases & Protein Names"):
                    display_alias_table(alias_info)

            # --- Source articles ---
            with st.expander(f"Source Articles ({len(articles)})"):
                display_articles(articles)

        # --- Download buttons (full report covering all genes) ---
        successful = [r for r in results if r["summary"]]
        if successful:
            st.divider()

            genes_label = ", ".join(r["canonical"] or r["gene"] for r in successful)
            file_stem = "_".join(r["canonical"] or r["gene"] for r in successful)
            file_stem = f"{file_stem}_{cell_type.strip().replace(' ', '_')}"

            # Build markdown report
            markdown_report = f"# Gene-Cell Function Report\n"
            markdown_report += f"**Genes:** {genes_label}\n"
            markdown_report += f"**Cell Type:** {cell_type.strip()}\n"
            markdown_report += f"**Generated:** {timestamp}\n\n"

            for r in successful:
                gene_name = r["canonical"] or r["gene"]
                markdown_report += f"## {gene_name}\n\n"
                markdown_report += f"{r['summary']}\n\n"
                if r["alias_info"]:
                    markdown_report += f"### Aliases\n\n{build_alias_table_md(r['alias_info'])}\n"
                markdown_report += f"### Source Articles\n\n{articles_to_md(r['articles'])}"

            # Build HTML report
            html_title = f"Gene-Cell Function Report: {genes_label} in {cell_type.strip()}"
            html_report = HTML_HEADER.format(title=html_title)
            html_report += f"    <h1>Gene-Cell Function Report</h1>\n"
            html_report += f'    <div class="meta">\n'
            html_report += f"        <strong>Genes:</strong> {genes_label}<br>\n"
            html_report += f"        <strong>Cell Type:</strong> {cell_type.strip()}<br>\n"
            html_report += f"        <strong>Generated:</strong> {timestamp}\n"
            html_report += f"    </div>\n"

            for r in successful:
                gene_name = r["canonical"] or r["gene"]
                html_report += f'    <div class="gene-section">\n'
                html_report += f"    <h2>{gene_name}</h2>\n"
                html_report += f"    <div>{r['summary'].replace(chr(10), '<br>')}</div>\n"
                if r["alias_info"]:
                    html_report += f"    <h3>Aliases</h3>\n{build_alias_table_html(r['alias_info'])}"
                html_report += f"    <h3>Source Articles</h3>\n{articles_to_html(r['articles'])}"
                html_report += f"    </div>\n"

            html_report += HTML_FOOTER

            dl1, dl2 = st.columns(2)
            with dl1:
                st.download_button(
                    label="Download as Markdown",
                    data=markdown_report,
                    file_name=f"{file_stem}_report.md",
                    mime="text/markdown",
                )
            with dl2:
                st.download_button(
                    label="Download as HTML",
                    data=html_report,
                    file_name=f"{file_stem}_report.html",
                    mime="text/html",
                )
