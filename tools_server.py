import io, json, logging, os, re, sqlite3, time
import xml.etree.ElementTree as ET
from difflib import SequenceMatcher
from pathlib import Path

import arxiv
import httpx
from pypdf import PdfReader
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("research-tools")
arxiv_client = arxiv.Client(page_size=5)
http = httpx.Client(timeout=30, follow_redirects=True, headers={"User-Agent": "research-assistant/0.1"})
logging.getLogger("httpx").setLevel(logging.WARNING)

DB = sqlite3.connect(Path(__file__).parent / "cache.db", timeout=10, check_same_thread=False)
DB.execute("CREATE TABLE IF NOT EXISTS search_cache (key TEXT PRIMARY KEY, value TEXT, ts REAL)")
DB.execute("CREATE TABLE IF NOT EXISTS notes (topic TEXT, note TEXT, ts REAL)")
CACHE_TTL = 7 * 24 * 3600
MAX_RESULTS = 5
ABSTRACT_CHARS = 1200        # keep tool output small enough for a local model's context
CHUNK_CHARS = 1500


def cached(key: str, fetch) -> str:
    """Return a cached result for `key`, or call fetch() and cache it. Errors are returned, not cached."""
    row = DB.execute("SELECT value, ts FROM search_cache WHERE key=?", (key,)).fetchone()
    if row and time.time() - row[1] < CACHE_TTL:
        return row[0]                                   # cache hit, no API call
    try:
        result = fetch()
    except Exception as e:                              # return the error so the agent can try something else
        return f"Tool error: {type(e).__name__}: {e}"
    DB.execute("INSERT OR REPLACE INTO search_cache VALUES (?,?,?)", (key, result, time.time()))
    DB.commit()
    return result


def get(url: str, retries: int = 3, **kwargs) -> httpx.Response:
    """GET with backoff on rate limits (Semantic Scholar's keyless tier often returns 429)."""
    for attempt in range(retries):
        r = http.get(url, **kwargs)
        if r.status_code != 429:
            break
        time.sleep(2 ** (attempt + 1))
    return r


def short(text: str | None) -> str:
    text = " ".join((text or "").split())
    return text[:ABSTRACT_CHARS] + ("..." if len(text) > ABSTRACT_CHARS else "")


# ---------- Search tools ----------

@mcp.tool()
def arxiv_search(query: str) -> str:
    """Search arXiv for preprints (strong for ML/CS). Returns arXiv id, title, authors, year, link and abstract."""
    def fetch():
        # arXiv's free-text search matches any word anywhere (even author names), so require every keyword
        words = re.findall(r"[\w-]+", query)
        search = query if ":" in query else " AND ".join(f"all:{w}" for w in words)
        results = arxiv_client.results(arxiv.Search(query=search, max_results=MAX_RESULTS))
        return "\n\n".join(
            f"arXiv id: {r.get_short_id()}\nTitle: {r.title}\n"
            f"Authors: {', '.join(a.name for a in r.authors[:5])}\nYear: {r.published.year}\n"
            f"Link: {r.entry_id}\nAbstract: {short(r.summary)}"
            for r in results
        ) or "No results."
    return cached(f"arxiv:{query.strip().lower()}", fetch)


@mcp.tool()
def semantic_scholar_search(query: str) -> str:
    """Search Semantic Scholar across all fields and publishers. Returns title, authors, year, venue,
    citation count, link and abstract. Use citation counts to find the most influential papers."""
    def fetch():
        headers = {"x-api-key": os.environ["S2_API_KEY"]} if os.environ.get("S2_API_KEY") else {}
        r = get("https://api.semanticscholar.org/graph/v1/paper/search", headers=headers, params={
            "query": query, "limit": MAX_RESULTS,
            "fields": "title,year,authors,venue,citationCount,abstract,url,externalIds",
        })
        r.raise_for_status()
        papers = r.json().get("data", [])
        return "\n\n".join(
            f"Title: {p['title']}\nAuthors: {', '.join(a['name'] for a in (p.get('authors') or [])[:5])}\n"
            f"Year: {p.get('year')}\nVenue: {p.get('venue') or 'n/a'}\nCitations: {p.get('citationCount')}\n"
            f"arXiv id: {(p.get('externalIds') or {}).get('ArXiv', 'n/a')}\n"
            f"Link: {p.get('url')}\nAbstract: {short(p.get('abstract')) or 'n/a'}"
            for p in papers
        ) or "No results."
    return cached(f"s2:{query.strip().lower()}", fetch)


@mcp.tool()
def pubmed_search(query: str) -> str:
    """Search PubMed for biomedical and clinical papers (journals, not preprints).
    Returns PMID, title, authors, journal, year, link and abstract."""
    def fetch():
        eutils = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        key = {"api_key": os.environ["NCBI_API_KEY"]} if os.environ.get("NCBI_API_KEY") else {}
        r = http.get(f"{eutils}/esearch.fcgi", params={"db": "pubmed", "term": query, "retmax": MAX_RESULTS,
                                                         "retmode": "json", "sort": "relevance", **key})
        r.raise_for_status()
        ids = r.json()["esearchresult"]["idlist"]
        if not ids:
            return "No results."
        r = http.get(f"{eutils}/efetch.fcgi", params={"db": "pubmed", "id": ",".join(ids), "retmode": "xml", **key})
        r.raise_for_status()
        out = []
        for art in ET.fromstring(r.content).iter("PubmedArticle"):
            pmid = art.findtext(".//PMID")
            authors = [f"{a.findtext('LastName')} {a.findtext('Initials') or ''}".strip()
                       for a in art.iter("Author") if a.findtext("LastName")]
            abstract = " ".join("".join(t.itertext()) for t in art.iter("AbstractText"))
            out.append(
                f"PMID: {pmid}\nTitle: {''.join(art.find('.//ArticleTitle').itertext())}\n"
                f"Authors: {', '.join(authors[:5])}\nJournal: {art.findtext('.//Journal/Title')}\n"
                f"Year: {art.findtext('.//JournalIssue/PubDate/Year') or art.findtext('.//PubDate/MedlineDate')}\n"
                f"Link: https://pubmed.ncbi.nlm.nih.gov/{pmid}/\nAbstract: {short(abstract) or 'n/a'}"
            )
        return "\n\n".join(out)
    return cached(f"pubmed:{query.strip().lower()}", fetch)


# ---------- Full-text retrieval ----------

def paper_text(arxiv_id: str) -> str:
    """Download an arXiv PDF and extract its text (cached)."""
    def fetch():
        r = http.get(f"https://arxiv.org/pdf/{arxiv_id}")
        r.raise_for_status()
        return "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(r.content)).pages)
    return cached(f"pdf:{arxiv_id}", fetch)


@mcp.tool()
def read_paper(arxiv_id: str, question: str) -> str:
    """Read the full text of an arXiv paper and return the passages most relevant to `question`.
    Use it on the most important papers to get methods, datasets and results, not just the abstract."""
    arxiv_id = re.sub(r"^(https?://arxiv\.org/(abs|pdf)/|arXiv:)", "", arxiv_id.strip()).removesuffix(".pdf")
    text = paper_text(arxiv_id)
    if text.startswith("Tool error"):
        return text
    chunks = [text[i:i + CHUNK_CHARS] for i in range(0, len(text), CHUNK_CHARS - 200)]   # 200-char overlap
    terms = {w for w in re.findall(r"[a-z0-9]+", question.lower()) if len(w) > 3}
    scored = sorted(chunks, key=lambda c: sum(c.lower().count(t) for t in terms), reverse=True)
    passages = "\n\n---\n\n".join(" ".join(c.split()) for c in scored[:3])
    return f"Passages from arXiv:{arxiv_id} (https://arxiv.org/abs/{arxiv_id}):\n\n{passages}"


# ---------- Citation check ----------

@mcp.tool()
def verify_citation(title: str) -> str:
    """Check that a paper with this title really exists. Returns the closest match with its year, authors and link,
    or says it was not found. Use it before citing a paper you did not get from a search tool in this session."""
    def fetch():
        r = get("https://api.semanticscholar.org/graph/v1/paper/search/match",
                     params={"query": title, "fields": "title,year,authors,url"})
        if r.status_code == 404:
            return json.dumps({})
        r.raise_for_status()
        return json.dumps(r.json()["data"][0])
    raw = cached(f"match:{title.strip().lower()}", fetch)
    if raw.startswith("Tool error"):
        return raw
    match = json.loads(raw)
    similarity = SequenceMatcher(None, title.lower(), (match.get("title") or "").lower()).ratio()
    if not match or similarity < 0.85:
        return f"NOT FOUND: no paper titled '{title}'. Do not cite it."
    authors = ", ".join(a["name"] for a in match.get("authors", [])[:3])
    return f"VERIFIED: {match['title']} ({match.get('year')}), {authors}. Link: {match.get('url')}"


# ---------- Memory ----------

@mcp.tool()
def save_note(topic: str, note: str) -> str:
    """Save a key finding (paper + claim) to memory so other researchers can reuse it."""
    DB.execute("INSERT INTO notes VALUES (?,?,?)", (topic, note, time.time()))
    DB.commit()
    return "saved"


@mcp.tool()
def recall_notes(topic: str) -> str:
    """Recall findings previously saved about a topic. Check this before searching."""
    rows = DB.execute("SELECT note FROM notes WHERE topic LIKE ?", (f"%{topic}%",)).fetchall()
    return "\n".join(r[0] for r in rows) or "No notes found."


if __name__ == "__main__":
    mcp.run(transport="stdio")
