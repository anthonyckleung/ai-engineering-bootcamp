from fastmcp import FastMCP
from src.items_mcp_server.utils import retrieve_context, process_context
from typing import Literal, Dict, Any, Annotated, List, Optional

mcp = FastMCP("items")


@mcp.tool()
def get_formatted_context(query: str, top_k: int = 5) -> Dict[str, Any]:

    """Get the top k context, each representing an job posting for a given query.
    
    Args:
        query: The query to get the top k context for
        top_k: The number of context chunks to retrieve, works best with 5 or more
    
    Returns:
        A dictionary of the top k context chunks with IDs prepending each chunk, each representing job posting for a given query.
    """

    context = retrieve_context(query, top_k)
    # formatted_context = process_context(context)

    return context


if __name__ == "__main__":
    mcp.run(transport='http', host="0.0.0.0", port=8000)