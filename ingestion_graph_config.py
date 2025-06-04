from langgraph.graph import StateGraph, END
from state import IngestionState

# Import your nodes
from nodes.ingestion_nodes.document_loader import load_document
from nodes.ingestion_nodes.content_classifier import content_classifier
from nodes.ingestion_nodes.text_chunker import text_chunker
from nodes.ingestion_nodes.table_chunker import table_chunker
from nodes.ingestion_nodes.visual_chunker import visual_chunker
from nodes.ingestion_nodes.chunk_aggregator import chunk_aggregator
from nodes.ingestion_nodes.embedding_node import embedding_node

# Build LangGraph
graph = StateGraph(IngestionState)

# Add nodes
graph.add_node("DocumentLoader", load_document)
graph.add_node("ContentClassifier", content_classifier)
graph.add_node("TextChunker", text_chunker)
graph.add_node("TableChunker", table_chunker)
graph.add_node("VisualChunker", visual_chunker)
graph.add_node("ChunkAggregator", chunk_aggregator)
graph.add_node("EmbeddingNode", embedding_node)

# Entry point
graph.set_entry_point("DocumentLoader")
graph.add_edge("DocumentLoader", "ContentClassifier")

# ContentClassifier branching logic
def classifier_branch(state: IngestionState):
    if not (state.has_text or state.has_table or state.has_visual):
        return END
    elif state.has_text:
        return "TextChunker"
    elif state.has_table:
        return "TableChunker"
    else:
        return "VisualChunker"

graph.add_conditional_edges("ContentClassifier", classifier_branch, {
    "TextChunker": "TextChunker",
    "TableChunker": "TableChunker",
    "VisualChunker": "VisualChunker",
    END: END
})

# After TextChunker → check for table or visual or end
def text_branch(state: IngestionState):
    if state.has_table:
        return "TableChunker"
    elif state.has_visual:
        return "VisualChunker"
    else:
        return END

graph.add_conditional_edges("TextChunker", text_branch, {
    "TableChunker": "TableChunker",
    "VisualChunker": "VisualChunker",
    END: END
})

# After TableChunker → check for visual or end
def table_branch(state: IngestionState):
    if state.has_visual:
        return "VisualChunker"
    else:
        return END

graph.add_conditional_edges("TableChunker", table_branch, {
    "VisualChunker": "VisualChunker",
    END: END
})

# After VisualChunker → go to ChunkAggregator
graph.add_edge("VisualChunker", "ChunkAggregator")

# After ChunkAggregator → EmbeddingNode
graph.add_edge("ChunkAggregator", "EmbeddingNode")

# After EmbeddingNode → END
graph.add_edge("EmbeddingNode", END)

# Compile graph
ingestion_flow = graph.compile()
