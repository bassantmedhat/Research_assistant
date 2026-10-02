import sqlite3, time
from mcp.server.fastmcp import FastMCP
from langchain_community.utilities import ArxivAPIWrapper

mcp = FastMCP("research-tools")
arxiv = ArxivAPIWrapper(top_k_results=5, doc_content_chars_max=4000)

DB = sqlite3.connect("cache.db", timeout=10, check_same_thread=False)
DB.execute("CREATE TABLE IF NOT EXISTS search_cache (key TEXT PRIMARY KEY, value TEXT, ts REAL)")
DB.execute("CREATE TABLE IF NOT EXISTS notes (topic TEXT, note TEXT, ts REAL)")
CACHE_TTL = 7 * 24 * 3600


@mcp.tool()
def archiv_search(query: str):
    """Search arXiv for academic papers. Returns titles, authors, dates and abstracts."""
    key = query.strip().lower()
    row = DB.execute("SELECT value, ts FROM search_cache WHERE key=?", (key,)).fetchone()
    if row and time.time() - row[1] < CACHE_TTL:
        return row[0]                                   # cache hit, no API call
    result = arxiv.run(query)
    DB.execute("INSERT OR REPLACE INTO search_cache VALUES (?,?,?)", (key, result, time.time()))
    DB.commit()
    return result


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