"""Contextual relevance filtering for gene-cell function articles.

Uses LLM to assess whether an abstract describes gene function IN a specific
cell type (not just co-occurring mentions).
"""

import json
from dataclasses import dataclass
from typing import List
import ollama

from pubmed import Article


class ContextFilterError(Exception):
    """Raised when relevance filtering fails."""


@dataclass
class RelevanceResult:
    """Result of relevance assessment for a single article."""
    article: Article
    category: str  # 'DIRECT_FUNCTION', 'INDIRECT_MENTION', 'UNCERTAIN'
    confidence: float  # 0.0-1.0
    reasoning: str
    relevant: bool


def assess_relevance(
    gene: str,
    cell_type: str,
    article: Article,
    model: str = "llama3.2:latest",
    include_borderline: bool = False,
) -> RelevanceResult:
    """Assess whether an article describes gene function in the specified cell type.

    Args:
        gene: Gene symbol (canonical)
        cell_type: Cell type to evaluate context for
        article: PubMed article with abstract
        model: Ollama model to use
        include_borderline: If True, include INDIRECT_MENTION with high confidence

    Returns:
        RelevanceResult with classification and relevance flag
    """
    if not article.abstract or not article.abstract.strip():
        return RelevanceResult(
            article=article,
            category="UNCERTAIN",
            confidence=0.0,
            reasoning="No abstract available",
            relevant=False,
        )

    prompt = _build_assessment_prompt(gene, cell_type, article)

    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            format="json",
        )
        content = response["message"]["content"]
        result = json.loads(content)

        category = result.get("category", "UNCERTAIN")
        confidence = float(result.get("confidence", 0.0))
        reasoning = result.get("explanation", result.get("reasoning", ""))

        # Determine relevance based on category and settings
        if category == "DIRECT_FUNCTION":
            relevant = confidence >= 0.6
        elif category == "INDIRECT_MENTION":
            relevant = include_borderline and confidence >= 0.8
        else:  # UNCERTAIN
            relevant = False

        return RelevanceResult(
            article=article,
            category=category,
            confidence=confidence,
            reasoning=reasoning,
            relevant=relevant,
        )

    except json.JSONDecodeError as e:
        raise ContextFilterError(f"Failed to parse LLM response as JSON: {e}") from e
    except ollama.ResponseError as e:
        raise ContextFilterError(f"Ollama error: {e}") from e
    except Exception as e:
        raise ContextFilterError(f"Relevance assessment failed: {e}") from e


def batch_assess_relevance(
    gene: str,
    cell_type: str,
    articles: List[Article],
    model: str = "llama3.2:latest",
    include_borderline: bool = False,
) -> List[RelevanceResult]:
    """Assess relevance for multiple articles in a single LLM call.

    More efficient than individual calls for large article sets.
    """
    articles_with_abstracts = [a for a in articles if a.abstract and a.abstract.strip()]

    if not articles_with_abstracts:
        return []

    prompt = _build_batch_prompt(gene, cell_type, articles_with_abstracts)

    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            format="json",
        )
        content = response["message"]["content"]
        results_data = json.loads(content)

        # Handle both single result and array of results
        if isinstance(results_data, dict) and "assessments" in results_data:
            assessments = results_data["assessments"]
        elif isinstance(results_data, list):
            assessments = results_data
        else:
            assessments = [results_data]

        results = []
        for i, article in enumerate(articles_with_abstracts):
            if i < len(assessments):
                data = assessments[i]
                category = data.get("category", "UNCERTAIN")
                confidence = float(data.get("confidence", 0.0))
                reasoning = data.get("explanation", data.get("reasoning", ""))

                if category == "DIRECT_FUNCTION":
                    relevant = confidence >= 0.6
                elif category == "INDIRECT_MENTION":
                    relevant = include_borderline and confidence >= 0.8
                else:
                    relevant = False

                results.append(RelevanceResult(
                    article=article,
                    category=category,
                    confidence=confidence,
                    reasoning=reasoning,
                    relevant=relevant,
                ))
            else:
                # Missing assessment for this article
                results.append(RelevanceResult(
                    article=article,
                    category="UNCERTAIN",
                    confidence=0.0,
                    reasoning="No assessment returned",
                    relevant=False,
                ))

        return results

    except json.JSONDecodeError as e:
        raise ContextFilterError(f"Failed to parse batch LLM response: {e}") from e
    except Exception as e:
        raise ContextFilterError(f"Batch assessment failed: {e}") from e


def _build_assessment_prompt(gene: str, cell_type: str, article: Article) -> str:
    """Build prompt for single article relevance assessment."""
    return f"""You are evaluating whether a research article describes the function of a gene specifically in a particular cell type.

Gene: {gene}
Cell Type: {cell_type}

Article:
Title: {article.title}
Abstract: {article.abstract}

Classify this article into ONE category:

A. DIRECT_FUNCTION - The gene's molecular function, expression, regulation, or biological role is explicitly described IN the specified cell type. Examples:
   - "BRCA1 expression in T-cells mediates..."
   - "FOXP3 functions as a transcription factor in regulatory T cells..."
   - "CD8+ T cells require BRCA1 for DNA repair during proliferation..."

B. INDIRECT_MENTION - Both terms appear but the gene's function IN the specified cell type is not discussed. Examples:
   - "BRCA1 mutations in breast cancer... T-cells were used as controls"
   - "T-cell infiltration in BRCA1-mutant tumors..." (BRCA1 is in tumor, not T-cells)
   - "We compared BRCA1 expression in cancer cells vs normal cells..." (no T-cell function)

C. UNCERTAIN - Cannot determine from the abstract whether the gene's function in the cell type is discussed.

Respond with a JSON object:
{{
  "category": "A" or "B" or "C",
  "confidence": 0.0 to 1.0 (your certainty in this classification),
  "explanation": "Brief explanation of your reasoning (1-2 sentences)"
}}

Use category A only when the abstract explicitly links {gene} function TO {cell_type}."""


def _build_batch_prompt(gene: str, cell_type: str, articles: List[Article]) -> str:
    """Build prompt for batch relevance assessment."""
    articles_text = ""
    for i, article in enumerate(articles, 1):
        articles_text += f"\n\n--- ARTICLE {i} ---\n"
        articles_text += f"PMID: {article.pmid}\n"
        articles_text += f"Title: {article.title}\n"
        articles_text += f"Abstract: {article.abstract}\n"

    return f"""You are evaluating whether research articles describe the function of a gene specifically in a particular cell type.

Gene: {gene}
Cell Type: {cell_type}

For each article below, classify it into ONE category:

A. DIRECT_FUNCTION - The gene's molecular function, expression, regulation, or biological role is explicitly described IN the specified cell type.

B. INDIRECT_MENTION - Both terms appear but the gene's function IN the specified cell type is not discussed.

C. UNCERTAIN - Cannot determine from the abstract.

Articles:{articles_text}

Respond with a JSON object containing an "assessments" array with one entry per article, in the same order:
{{
  "assessments": [
    {{
      "category": "A" or "B" or "C",
      "confidence": 0.0 to 1.0,
      "explanation": "Brief reasoning"
    }},
    ... (one entry per article)
  ]
}}

Use category A only when the article explicitly links {gene} function TO {cell_type}."""


def filter_articles(
    results: List[RelevanceResult],
    min_confidence: float = 0.6,
) -> tuple[List[Article], List[RelevanceResult]]:
    """Split results into relevant articles and excluded results.

    Returns:
        Tuple of (relevant_articles, excluded_results)
    """
    relevant = [r.article for r in results if r.relevant and r.confidence >= min_confidence]
    excluded = [r for r in results if not r.relevant or r.confidence < min_confidence]
    return relevant, excluded
