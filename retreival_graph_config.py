from langgraph.graph import StateGraph, END
from state import QueryState

# Import nodes
from nodes.retrieval_nodes.index_loader import index_loader
from nodes.retrieval_nodes.query_classifier import query_classifier
from nodes.retrieval_nodes.query_embedder import query_embedder
from nodes.retrieval_nodes.retreiver import retriever
from nodes.retrieval_nodes.highlight_node import highlight_node
from nodes.retrieval_nodes.answer_generator import answer_generator
from nodes.retrieval_nodes.plotting_node import plotting_node
from nodes.retrieval_nodes.pandas_node import pandas_node
from nodes.retrieval_nodes.query_modality_classifier import query_modality_classifier

# Build LangGraph
graph = StateGraph(QueryState)

# Add nodes
graph.add_node("IndexLoader", index_loader)
graph.add_node("QueryClassifier", query_classifier)
graph.add_node("QueryEmbedder", query_embedder)
graph.add_node("Retriever", retriever)
graph.add_node("HighlightNode", highlight_node)
graph.add_node("AnswerGenerator", answer_generator)
graph.add_node("PlottingNode", plotting_node)
graph.add_node("PandasNode", pandas_node)
graph.add_node("QueryModalityClassifier", query_modality_classifier)

# Set entry point
graph.set_entry_point("IndexLoader")
graph.add_edge("IndexLoader", "QueryClassifier")
graph.add_edge("QueryClassifier", "QueryModalityClassifier")
graph.add_edge("QueryModalityClassifier", "QueryEmbedder")
graph.add_edge("QueryEmbedder", "Retriever")
graph.add_edge("Retriever", "HighlightNode")
graph.add_edge("HighlightNode", "AnswerGenerator")

# Conditional branching after AnswerGenerator
def enrichment_branch(state: QueryState):
    if state.plot_requested:
        return "PlottingNode"
    elif state.table_requested:
        return "PandasNode"
    else:
        return END

graph.add_conditional_edges("AnswerGenerator", enrichment_branch, {
    "PlottingNode": "PlottingNode",
    "PandasNode": "PandasNode",
    END: END
})

# After enrichment nodes → END
graph.add_edge("PlottingNode", END)
graph.add_edge("PandasNode", END)

# Compile LangGraph
query_flow = graph.compile()