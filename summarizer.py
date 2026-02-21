"""LLM summarization module using Ollama for gene-cell function summaries."""

import ollama


class SummarizerError(Exception):
    """Raised when LLM summarization fails."""


# Length presets: (label, paragraph instruction, detail level)
LENGTH_PRESETS = {
    "short": "Keep it to 1-2 short paragraphs. Be concise and direct.",
    "medium": "Be 2-4 paragraphs of integrated biological insight.",
    "long": "Provide a comprehensive 5-7 paragraph analysis with detailed discussion of mechanisms, evidence, and clinical significance.",
}


def summarize_gene_function(
    gene: str,
    cell_type: str,
    articles: list,
    model: str = "llama3.2:latest",
    summary_length: str = "medium",
) -> str:
    """Summarize gene function in a cell type based on PubMed articles.

    Uses a local Ollama LLM to generate a summary citing PMIDs.
    Articles without abstracts are filtered out before prompting.

    Args:
        summary_length: One of "short", "medium", "long".
    """
    # Filter to articles that have abstracts
    articles_with_abstracts = [a for a in articles if a.abstract.strip()]
    if not articles_with_abstracts:
        raise SummarizerError("No articles with abstracts available for summarization.")

    length_instruction = LENGTH_PRESETS.get(summary_length, LENGTH_PRESETS["medium"])

    # Build the literature context
    literature = []
    for a in articles_with_abstracts:
        literature.append(
            f"PMID: {a.pmid}\n"
            f"Title: {a.title}\n"
            f"Authors: {', '.join(a.authors[:3])}{'...' if len(a.authors) > 3 else ''}\n"
            f"Journal: {a.journal} ({a.year})\n"
            f"Abstract: {a.abstract}\n"
        )

    literature_text = "\n---\n".join(literature)

    prompt = f"""You are a molecular biology expert. Your task is to synthesize information across multiple research articles to explain the biological function of the gene **{gene}** specifically in **{cell_type}**.

IMPORTANT: Do NOT summarize individual articles. Instead, integrate findings across all articles to answer these questions:

1. **What does {gene} do in {cell_type}?** Describe its molecular functions and biological roles.
2. **How does it work?** Explain signaling pathways, protein interactions, and molecular mechanisms.
3. **What happens when it's dysregulated?** Note disease associations, phenotypes, and clinical relevance.
4. **What's the evidence?** Support your statements with citations [PMID: XXXXX].

Your output should:
- Focus on the GENE'S FUNCTION in the CELL TYPE, not article-by-article summaries
- Synthesize findings across multiple studies
- Highlight consensus findings and note contradictions if present
- {length_instruction}
- Use precise scientific terminology

Literature:
{literature_text}

Write your integrated summary of {gene} function in {cell_type}:"""

    return _call_ollama(prompt, model)


def summarize_gene_function_brief(
    gene: str,
    cell_type: str,
    articles: list,
    model: str = "llama3.2:latest",
    summary_length: str = "short",
) -> str:
    """Generate a focused summary of gene function for multi-gene reports.

    Args:
        summary_length: One of "short", "medium", "long".
    """
    articles_with_abstracts = [a for a in articles if a.abstract.strip()]
    if not articles_with_abstracts:
        raise SummarizerError(f"No articles with abstracts available for {gene}.")

    length_instruction = LENGTH_PRESETS.get(summary_length, LENGTH_PRESETS["short"])

    literature = []
    for a in articles_with_abstracts:
        literature.append(
            f"PMID: {a.pmid}\n"
            f"Title: {a.title}\n"
            f"Abstract: {a.abstract}\n"
        )

    literature_text = "\n---\n".join(literature)

    prompt = f"""You are a molecular biology expert. Write a summary of the function of the gene **{gene}** in **{cell_type}**.

Focus on:
- The primary biological function of {gene} in {cell_type}
- Key pathways or mechanisms
- Major disease associations (if any)
- Cite key articles by PMID [PMID: XXXXX]

{length_instruction}

Literature:
{literature_text}

Summary of {gene} in {cell_type}:"""

    return _call_ollama(prompt, model)


def _call_ollama(prompt: str, model: str) -> str:
    """Call Ollama chat API and return response content."""
    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response["message"]["content"]
    except ollama.ResponseError as e:
        raise SummarizerError(f"Ollama error: {e}") from e
    except Exception as e:
        raise SummarizerError(f"Failed to generate summary: {e}") from e
