# so we are basically here applying the section-wise chunking technique

# this is the same kind of problem as for the research paper problems 

import re

# Regex that matches GrantMatch's field labels at the start of a line.
# Examples it catches: "Program:", "Source:", "Eligibility:",
# "Eligibility (Cal Grant A):", "General requirements (both):", "Award:"
HEADER = re.compile(
    r"^(Program|Source"
    r"|Eligibility(?:\s*\([^)]*\))?"          # optional qualifier, e.g. "(Cal Grant A)"
    r"|General\s+Requirements(?:\s*\([^)]*\))?"
    r"|Award)\s*:",
    re.IGNORECASE | re.MULTILINE,
)

def _recursive_resplit(text, max_chars, overlap):

    """
    Break a long section into smaller pieces that fit within max_chars.
    Tries natural break points in order: paragraph -> line -> sentence -> word.

    if none works cleanly, falls back to fixed-width sliding windows.

    Args:
        text: A section that is too long to keep as one chunk.
        max_chars: Target maximum length for each sub chunk.
        overlap: Characters to repeat between consecutive sub-chunks.



    Returns:
        List of text strings.
    """

    if len(text) <= max_chars:
        return [text]
    
    # defining separators to make better chunks

    separators = ["\n\n", "\n", ". ", " "]

    for sep in separators:
        parts = text.split(sep)
        if len(parts) > 1:
            chunks, current = [], ""
            for part in parts:
                candidate = part if not current else current + sep + part
                if len(candidate) <= max_chars:
                    current = candidate
                else:
                    if current:
                        chunks.append(current)
                    current = part

            if current:
                chunks.append(current)

            
            # Accept this split only if no chunk is wildly over the limit (120%).

            if all(len(c) <= max_chars * 1.2 for c in chunks):
                result = []

                for i, c in enumerate(chunks):
                    if i > 0 and overlap > 0:
                        if i > 0 and overlap > 0:

                            # Prepend overlap from the previous sub-chunk:
                             c = chunks[i - 1][-overlap:] + c
                        result.append(c)

                return result
            

    return [text[i:i + max_chars] for i in range(0, len(text), max_chars - overlap)]
        

def chunk(pages, chunk_size=800, chunk_overlap=80):
    """
    Split research papers into section-based chunks.

    Args:
        pages: List of page dicts from the PDF loader.
        chunk_size: Used only when re-splitting very long sections.
        chunk_overlap: Overlap applied during re-splitting of long sections.

    Returns:
        List of {"text": str, "metadata": dict} chunk dicts.
        Metadata includes "section" (e.g. "Introduction", "Methods").
    """
    # Group all pages by PDF file path so we can join them into one document.
    by_source = {}
    for p in pages:
        by_source.setdefault(p["metadata"]["source"], []).append(p["page_content"])

    # Only re-split sections longer than this threshold.
    # Short-to-medium sections (Abstract, Conclusions, Background) stay whole —
    # that's the point of section-aware chunking.
    # Trade-off: the embedder truncates past ~256 tokens (~1200 chars), so
    # very large chunks lose tail content in embedding space, but the full
    # text is still available when the LLM generates an answer.
    resplit_threshold = max(chunk_size * 3, 2500)

    results = []
    for source, page_texts in by_source.items():
        # Stitch all pages of this paper into one continuous string.
        full_text = "\n".join(page_texts)

        # Find every section header and its position in the text.
        matches = list(HEADER.finditer(full_text))
        starts = [0] + [m.start() for m in matches]
        names = ["Front Matter"] + [m.group(1).title() for m in matches]

        # Walk each section from its start to the next section's start.
        for name, start, end in zip(names, starts, starts[1:] + [len(full_text)]):
            section_text = full_text[start:end].strip()
            if len(section_text) < 50:
                continue  # Skip empty or nearly-empty sections.

            if len(section_text) <= resplit_threshold:
                # Section is short enough — keep it as one chunk.
                results.append({
                    "text": section_text,
                    "metadata": {
                        "source": source,
                        "section": name,
                        "chunker": "section_wise",
                    },
                })
            else:
                # Section is very long — break it into smaller sub-chunks.
                for piece in _recursive_resplit(section_text, chunk_size, chunk_overlap):
                    piece = piece.strip()
                    if len(piece) < 20:
                        continue
                    results.append({
                        "text": piece,
                        "metadata": {
                            "source": source,
                            "section": name,  # Same section name on all sub-chunks.
                            "chunker": "section_wise",
                        },
                    })

    return results
