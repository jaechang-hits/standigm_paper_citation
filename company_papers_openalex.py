"""
Company Papers Comparison Tool using OpenAlex API

This tool fetches and compares academic papers from different companies
based on their affiliations using the OpenAlex API.
"""

import requests
import pandas as pd
from datetime import datetime
import time
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np


# Constants
OPENALEX_API_BASE = "https://api.openalex.org"
REQUEST_DELAY = 0.5  # seconds between requests (polite pool)


def search_institution(company_name: str, email: str) -> Optional[dict]:
    """
    Search for an institution in OpenAlex by name.

    Args:
        company_name: Name of the company/institution to search
        email: Email for polite pool access

    Returns:
        Dictionary with institution info or None if not found
    """
    url = f"{OPENALEX_API_BASE}/institutions"
    params = {
        "search": company_name,
        "mailto": email
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        if data.get("results") and len(data["results"]) > 0:
            inst = data["results"][0]
            return {
                "id": inst.get("id"),
                "display_name": inst.get("display_name"),
                "ror": inst.get("ror"),
                "country_code": inst.get("country_code"),
                "type": inst.get("type"),
                "works_count": inst.get("works_count"),
                "cited_by_count": inst.get("cited_by_count")
            }
        return None
    except requests.RequestException as e:
        print(f"Error searching institution '{company_name}': {e}")
        return None


def fetch_papers_by_institution(institution_id: str, email: str, max_papers: int = 500) -> list[dict]:
    """
    Fetch papers from a specific institution using OpenAlex API.

    Args:
        institution_id: OpenAlex institution ID
        email: Email for polite pool access
        max_papers: Maximum number of papers to fetch

    Returns:
        List of paper dictionaries
    """
    papers = []
    cursor = "*"
    per_page = 100

    # Extract short ID from full URL if needed
    if institution_id.startswith("https://"):
        institution_id = institution_id.split("/")[-1]

    while len(papers) < max_papers:
        url = f"{OPENALEX_API_BASE}/works"
        params = {
            "filter": f"institutions.id:{institution_id}",
            "sort": "cited_by_count:desc",
            "per-page": per_page,
            "cursor": cursor,
            "mailto": email
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            results = data.get("results", [])
            if not results:
                break

            for work in results:
                paper = _extract_paper_data(work)
                papers.append(paper)

                if len(papers) >= max_papers:
                    break

            # Get next cursor
            cursor = data.get("meta", {}).get("next_cursor")
            if not cursor:
                break

            time.sleep(REQUEST_DELAY)

        except requests.RequestException as e:
            print(f"Error fetching papers: {e}")
            break

    return papers


def fetch_papers_by_affiliation_text(company_name: str, email: str, max_papers: int = 500) -> list[dict]:
    """
    Fetch papers by searching raw affiliation strings.
    Useful for startups or companies without a registered institution ID.

    Args:
        company_name: Company name to search in affiliations
        email: Email for polite pool access
        max_papers: Maximum number of papers to fetch

    Returns:
        List of paper dictionaries
    """
    papers = []
    cursor = "*"
    per_page = 100

    while len(papers) < max_papers:
        url = f"{OPENALEX_API_BASE}/works"
        params = {
            "filter": f"raw_affiliation_strings.search:{company_name}",
            "sort": "cited_by_count:desc",
            "per-page": per_page,
            "cursor": cursor,
            "mailto": email
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            results = data.get("results", [])
            if not results:
                break

            for work in results:
                paper = _extract_paper_data(work)
                papers.append(paper)

                if len(papers) >= max_papers:
                    break

            # Get next cursor
            cursor = data.get("meta", {}).get("next_cursor")
            if not cursor:
                break

            time.sleep(REQUEST_DELAY)

        except requests.RequestException as e:
            print(f"Error fetching papers by affiliation: {e}")
            break

    return papers


def fetch_papers_by_author(author_id: str, email: str, max_papers: int = 500) -> list[dict]:
    """
    Fetch papers by author ID.

    Args:
        author_id: OpenAlex author ID (e.g., 'A5011252820')
        email: Email for polite pool access
        max_papers: Maximum number of papers to fetch

    Returns:
        List of paper dictionaries
    """
    papers = []
    cursor = "*"
    per_page = 100

    # Extract short ID from full URL if needed
    if author_id.startswith("https://"):
        author_id = author_id.split("/")[-1]

    while len(papers) < max_papers:
        url = f"{OPENALEX_API_BASE}/works"
        params = {
            "filter": f"author.id:{author_id}",
            "sort": "cited_by_count:desc",
            "per-page": per_page,
            "cursor": cursor,
            "mailto": email
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            results = data.get("results", [])
            if not results:
                break

            for work in results:
                paper = _extract_paper_data(work)
                papers.append(paper)

                if len(papers) >= max_papers:
                    break

            # Get next cursor
            cursor = data.get("meta", {}).get("next_cursor")
            if not cursor:
                break

            time.sleep(REQUEST_DELAY)

        except requests.RequestException as e:
            print(f"Error fetching papers by author: {e}")
            break

    return papers


def _extract_paper_data(work: dict) -> dict:
    """Extract relevant paper data from OpenAlex work object."""
    # Extract authors
    authors = []
    for authorship in work.get("authorships", []):
        author = authorship.get("author", {})
        if author.get("display_name"):
            authors.append(author["display_name"])

    # Extract venue/journal
    venue = None
    primary_location = work.get("primary_location", {})
    if primary_location:
        source = primary_location.get("source", {})
        if source:
            venue = source.get("display_name")

    # Extract OA status
    oa_info = work.get("open_access", {})
    oa_status = oa_info.get("oa_status", "unknown")

    return {
        "title": work.get("title", ""),
        "authors": "; ".join(authors),
        "year": work.get("publication_year"),
        "citations": work.get("cited_by_count", 0),
        "doi": work.get("doi"),
        "url": work.get("id"),
        "venue": venue,
        "oa_status": oa_status
    }


def calculate_h_index(citations: list[int]) -> int:
    """Calculate H-index from a list of citation counts."""
    sorted_citations = sorted(citations, reverse=True)
    h_index = 0
    for i, c in enumerate(sorted_citations):
        if c >= i + 1:
            h_index = i + 1
        else:
            break
    return h_index


def _normalize_title(title: str) -> str:
    """Normalize title for comparison."""
    if not title:
        return ""
    # Lowercase, remove punctuation and extra spaces
    import re
    normalized = title.lower()
    normalized = re.sub(r'[^\w\s]', '', normalized)
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    return normalized


def _titles_are_similar(title1: str, title2: str, threshold: float = 0.85) -> bool:
    """Check if two titles are similar enough to be considered duplicates."""
    norm1 = _normalize_title(title1)
    norm2 = _normalize_title(title2)

    # Exact match
    if norm1 == norm2:
        return True

    # Check if one is a prefix of the other (for truncated titles)
    min_len = min(len(norm1), len(norm2))
    if min_len > 30 and norm1[:min_len] == norm2[:min_len]:
        return True

    # Check word overlap ratio
    words1 = set(norm1.split())
    words2 = set(norm2.split())
    if not words1 or not words2:
        return False

    intersection = words1 & words2
    union = words1 | words2
    jaccard = len(intersection) / len(union)

    return jaccard >= threshold


def remove_duplicates(papers: list[dict]) -> list[dict]:
    """
    Remove duplicate papers based on title similarity.
    Keeps the version with the highest citation count.
    """
    if not papers:
        return papers

    # Sort by citations descending first (so we keep the most cited version)
    sorted_papers = sorted(papers, key=lambda p: p.get("citations", 0), reverse=True)

    unique_papers = []
    for paper in sorted_papers:
        is_duplicate = False
        for existing in unique_papers:
            if _titles_are_similar(paper.get("title", ""), existing.get("title", "")):
                is_duplicate = True
                break
        if not is_duplicate:
            unique_papers.append(paper)

    return unique_papers


def filter_by_keywords(papers: list[dict], keywords: list[str], exclude_keywords: list[str] = None) -> list[dict]:
    """
    Filter papers by keywords in title.

    Args:
        papers: List of paper dictionaries
        keywords: List of keywords to match (case-insensitive, any match)
        exclude_keywords: List of keywords to exclude (case-insensitive, any match)

    Returns:
        Filtered list of papers
    """
    if not keywords:
        return papers

    keywords_lower = [k.lower() for k in keywords]
    exclude_lower = [k.lower() for k in (exclude_keywords or [])]

    filtered = []
    for paper in papers:
        title = (paper.get("title") or "").lower()
        # Include if matches any keyword
        if any(kw in title for kw in keywords_lower):
            # Exclude if matches any exclude keyword
            if not any(ex in title for ex in exclude_lower):
                filtered.append(paper)
    return filtered


def compare_companies(company_list: list, email: str, max_papers: int = 500, min_year: int = None, keyword_filter: list[str] = None, exclude_keywords: list[str] = None) -> dict:
    """
    Compare papers from multiple companies.

    Args:
        company_list: List of company names (str) or tuples of (display_name, [search_terms])
                      Search terms can be:
                      - Regular text: searches affiliation strings
                      - "author:ID": searches by author ID (e.g., "author:A5011252820")
        email: Email for polite pool access
        max_papers: Maximum papers per company
        min_year: Minimum publication year (inclusive). Papers before this year are excluded.
        keyword_filter: List of keywords to filter papers by title (for author searches)
        exclude_keywords: List of keywords to exclude from results

    Returns:
        Dictionary with papers and comparison statistics
    """
    papers_dict = {}
    stats = {}

    for company_entry in company_list:
        # Support both simple string and (name, [search_terms]) format
        if isinstance(company_entry, tuple):
            company_name, search_terms = company_entry
        else:
            company_name = company_entry
            search_terms = [company_entry]

        print(f"\nSearching for: {company_name}")

        all_papers = []

        for term in search_terms:
            print(f"  Searching term: '{term}'")

            # Check if it's an author search
            if term.startswith("author:"):
                author_id = term.replace("author:", "")
                print(f"    Searching by author ID: {author_id}")
                papers = fetch_papers_by_author(author_id, email, max_papers)
                # Apply keyword filter for author searches
                if keyword_filter:
                    original = len(papers)
                    papers = filter_by_keywords(papers, keyword_filter, exclude_keywords)
                    print(f"    Keyword filtered: {original} -> {len(papers)} papers")
            else:
                # First try to find institution ID
                institution = search_institution(term, email)
                time.sleep(REQUEST_DELAY)

                if institution:
                    print(f"    Found institution: {institution['display_name']}")
                    papers = fetch_papers_by_institution(institution["id"], email, max_papers)
                else:
                    papers = fetch_papers_by_affiliation_text(term, email, max_papers)

            print(f"    Found {len(papers)} papers")
            all_papers.extend(papers)

        papers = all_papers

        # Filter by year if specified
        if min_year:
            original_count = len(papers)
            papers = [p for p in papers if p.get("year") and p.get("year") >= min_year]
            if original_count != len(papers):
                print(f"  Filtered {original_count - len(papers)} paper(s) before {min_year}")

        # Apply exclude keywords filter to all results
        if exclude_keywords:
            exclude_lower = [k.lower() for k in exclude_keywords]
            original_count = len(papers)
            papers = [p for p in papers if not any(ex in (p.get("title") or "").lower() for ex in exclude_lower)]
            if original_count != len(papers):
                print(f"  Excluded {original_count - len(papers)} paper(s) by keywords")

        # Remove duplicate papers (e.g., preprints and published versions)
        original_count = len(papers)
        papers = remove_duplicates(papers)
        if original_count != len(papers):
            print(f"  Removed {original_count - len(papers)} duplicate(s)")

        papers_dict[company_name] = papers

        # Calculate statistics
        if papers:
            citations = [p["citations"] for p in papers]
            stats[company_name] = {
                "total_papers": len(papers),
                "total_citations": sum(citations),
                "avg_citations": round(sum(citations) / len(papers), 2),
                "max_citations": max(citations),
                "h_index": calculate_h_index(citations)
            }
        else:
            stats[company_name] = {
                "total_papers": 0,
                "total_citations": 0,
                "avg_citations": 0,
                "max_citations": 0,
                "h_index": 0
            }

        print(f"  Found {len(papers)} papers")

    return {
        "papers": papers_dict,
        "stats": stats
    }


def save_results(papers_dict: dict, comparison_stats: dict, output_prefix: str = "") -> None:
    """
    Save results to CSV and Markdown files.

    Args:
        papers_dict: Dictionary of company -> papers list
        comparison_stats: Dictionary of comparison statistics
        output_prefix: Prefix for output filenames
    """
    # Save individual CSV files for each company
    for company, papers in papers_dict.items():
        if papers:
            df = pd.DataFrame(papers)
            filename = f"{output_prefix}{company.lower().replace(' ', '_')}_papers.csv"
            df.to_csv(filename, index=False, encoding='utf-8')
            print(f"Saved: {filename}")

    # Create comparison markdown report
    md_content = _generate_markdown_report(papers_dict, comparison_stats)
    md_filename = f"{output_prefix}company_comparison.md"
    with open(md_filename, 'w', encoding='utf-8') as f:
        f.write(md_content)
    print(f"Saved: {md_filename}")

    # Save combined Excel file
    try:
        excel_filename = f"{output_prefix}company_papers_combined.xlsx"
        with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
            for company, papers in papers_dict.items():
                if papers:
                    df = pd.DataFrame(papers)
                    sheet_name = company[:31]  # Excel sheet name limit
                    df.to_excel(writer, sheet_name=sheet_name, index=False)

            # Add stats sheet
            stats_df = pd.DataFrame(comparison_stats).T
            stats_df.to_excel(writer, sheet_name='Comparison Stats')
        print(f"Saved: {excel_filename}")
    except ImportError:
        print("Note: openpyxl not installed, skipping Excel output")

    # Generate comparison charts
    generate_comparison_charts(papers_dict, comparison_stats, output_prefix)


def generate_comparison_charts(papers_dict: dict, comparison_stats: dict, output_prefix: str = "") -> None:
    """
    Generate comparison bar charts for the companies.

    Args:
        papers_dict: Dictionary of company -> papers list
        comparison_stats: Dictionary of comparison statistics
        output_prefix: Prefix for output filename
    """
    plt.switch_backend('Agg')  # Use non-interactive backend
    plt.rcParams['font.family'] = 'AppleGothic'  # Korean font for macOS
    plt.rcParams['axes.unicode_minus'] = False

    companies = list(papers_dict.keys())
    if len(companies) < 2:
        print("Need at least 2 companies for comparison charts")
        return

    # Convert to DataFrames
    dfs = {company: pd.DataFrame(papers) for company, papers in papers_dict.items() if papers}

    # Reorder: HITS first (blue), then others
    ordered_companies = []
    for c in companies:
        if 'HITS' in c.upper():
            ordered_companies.insert(0, c)
        else:
            ordered_companies.append(c)

    # Colors: Blue for first (HITS), Orange for second
    colors = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63'][:len(companies)]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Get data in order
    total_citations = [dfs[c]['citations'].sum() for c in ordered_companies]
    avg_citations = [dfs[c]['citations'].mean() for c in ordered_companies]

    # 1. Total Citations (전체 인용수)
    ax1 = axes[0]
    bars1 = ax1.bar(ordered_companies, total_citations, color=colors, width=0.6)
    ax1.set_ylabel('인용수', fontsize=18)
    ax1.set_title('전체 인용수', fontsize=21, fontweight='bold')
    ax1.tick_params(axis='both', labelsize=15)
    for bar, val in zip(bars1, total_citations):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(total_citations)*0.02,
                 f'{int(val)}', ha='center', va='bottom', fontsize=18, fontweight='bold')
    ax1.set_ylim(0, max(total_citations) * 1.15)

    # 2. Average Citations per Paper (논문당 평균 인용수)
    ax2 = axes[1]
    bars2 = ax2.bar(ordered_companies, avg_citations, color=colors, width=0.6)
    ax2.set_ylabel('인용수', fontsize=18)
    ax2.set_title('논문당 평균 인용수', fontsize=21, fontweight='bold')
    ax2.tick_params(axis='both', labelsize=15)
    for bar, val in zip(bars2, avg_citations):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(avg_citations)*0.02,
                 f'{val:.1f}', ha='center', va='bottom', fontsize=18, fontweight='bold')
    ax2.set_ylim(0, max(avg_citations) * 1.15)

    # Add legend (HITS first)
    fig.legend([bars1[i] for i in range(len(ordered_companies))], ordered_companies,
               loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=len(ordered_companies), fontsize=16)

    plt.suptitle('HITS vs Standigm 논문 인용수 비교', fontsize=24, fontweight='bold', y=1.02)
    plt.tight_layout()

    chart_filename = f"{output_prefix}company_comparison.png"
    plt.savefig(chart_filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {chart_filename}")


def _generate_markdown_report(papers_dict: dict, comparison_stats: dict) -> str:
    """Generate a markdown comparison report."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    md = f"""# Company Papers Comparison Report

Generated: {now}

## Summary Statistics

| Company | Total Papers | Total Citations | Avg Citations | Max Citations | H-index |
|---------|-------------|-----------------|---------------|---------------|---------|
"""

    for company, stats in comparison_stats.items():
        md += f"| {company} | {stats['total_papers']} | {stats['total_citations']} | {stats['avg_citations']} | {stats['max_citations']} | {stats['h_index']} |\n"

    md += "\n## Top Cited Papers by Company\n"

    for company, papers in papers_dict.items():
        md += f"\n### {company}\n\n"
        if papers:
            top_papers = sorted(papers, key=lambda x: x['citations'], reverse=True)[:10]
            md += "| # | Title | Year | Citations | Venue |\n"
            md += "|---|-------|------|-----------|-------|\n"
            for i, paper in enumerate(top_papers, 1):
                title = paper['title'][:60] + "..." if len(paper['title']) > 60 else paper['title']
                title = title.replace("|", "\\|")  # Escape pipe characters
                venue = paper['venue'] or "N/A"
                venue = venue[:30] + "..." if len(venue) > 30 else venue
                venue = venue.replace("|", "\\|")
                md += f"| {i} | {title} | {paper['year']} | {paper['citations']} | {venue} |\n"
        else:
            md += "No papers found.\n"

    md += "\n## Data Source\n\nData retrieved from [OpenAlex](https://openalex.org) API.\n"

    return md


def main():
    """Main function to run the comparison."""
    # Configuration
    email = "user@example.com"  # Replace with your email for polite pool access

    # Drug discovery related keywords for filtering author searches
    drug_keywords = [
        "drug", "molecular", "protein", "ligand", "binding", "interaction",
        "generative", "autoencoder", "scaffold", "retrosynthesis", "synthetic",
        "affinity", "docking", "pharmacophore", "bioisoster", "ADMET",
        "PIGNet", "DFRscore", "HyperLab"
    ]

    # Keywords to exclude (not related to drug discovery)
    exclude_keywords = [
        "silk", "supramolecular", "photoresist", "fibrohexamerin",
        "thin film", "lithography"
    ]

    # Companies to compare - can use tuple format (display_name, [search_terms]) for multiple search terms
    # Use "author:ID" format to search by author ID
    companies = [
        "Standigm",
        ("HITS", [
            "HITS Incorporation",
            "HITS Inc Korea",
            "author:A5011252820"  # Jaechang Lim
        ])
    ]
    max_papers = 500
    min_year = 2018  # Only include papers from 2018 onwards

    print("=" * 60)
    print("Company Papers Comparison Tool (OpenAlex)")
    print("=" * 60)

    # Run comparison
    results = compare_companies(companies, email, max_papers, min_year=min_year, keyword_filter=drug_keywords, exclude_keywords=exclude_keywords)

    # Save results
    save_results(results["papers"], results["stats"])

    # Print summary
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)

    for company, stats in results["stats"].items():
        print(f"\n{company}:")
        print(f"  Total Papers: {stats['total_papers']}")
        print(f"  Total Citations: {stats['total_citations']}")
        print(f"  Average Citations: {stats['avg_citations']}")
        print(f"  Max Citations: {stats['max_citations']}")
        print(f"  H-index: {stats['h_index']}")


if __name__ == "__main__":
    main()
