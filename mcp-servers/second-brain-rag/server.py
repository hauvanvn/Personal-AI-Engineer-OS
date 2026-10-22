from mcp.server.fastmcp import FastMCP
from pydantic import Field
from typing import Optional
from milvus_store import SecondBrainStore

# init FastMCP Server
mcp = FastMCP("PaPa-RAG")
store = SecondBrainStore()

@mcp.tool()
def search_knowledge(
    query: str = Field(..., description="Query for document, slide or personal note"),
    doc_type: Optional[str] = Field(None, description="Filter type: 'note', 'slide', 'pdf', 'architecture'"),
    top_k: int = Field(4, description="Number of return results")
) -> str:
    """
    Query from Second Brain of user \n
    Alway call when user ask about old project, saved data,...
    """
    results = store.search_with_time_decay(query=query, doc_type=doc_type, top_k=top_k)
    if not results:
        return "No query matched in Second Brain."

    formatted = []
    for idx, item in enumerate(results, 1):
        formatted.append(
            f"[{idx}] Source: {item['source_file']} (Type: {item['doc_type']} Score: {item['score']})\n"
            f"Text: {item['text']}"
        )

    return "\n\n---\n\n".join(formatted)

@mcp.tool()
def save_memory_note(
    content: str = Field(..., description="Data need to save"),
    source_name: str = Field(..., description="Name of note, Example: 'architecture_decision_v1.md'"),
    doc_type: str = Field("note", description="Type: 'note', 'architecture', 'todo'")
) -> str:
    """Save a new note or important data in longtern memory (Vector DB)."""
    count = store.add_document(text=content, source_file=source_name, doc_type=doc_type)
    return f"Save successful {count} record into '{source_name}'."

if __name__ == "__main__":
    # Hỗ trợ chạy qua chuẩn stdio (local) hoặc sse (docker/microservice)
    mcp.run()