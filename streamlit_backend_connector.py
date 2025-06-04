from ingestion_graph_config import ingestion_flow
from retreival_graph_config import query_flow
from state import IngestionState, QueryState

def run_ingestion(file_path: str, vlm_prompt: str = "Describe the visual content"):
    """
    Trigger full ingestion pipeline on uploaded PDF.
    """
    # Build ingestion state
    state = IngestionState(
        file_path=file_path,
        index_folder="vector_store",  # or wherever you store vectors
        vlm_prompt="Analyze visual content of this page..."
    )

    # Run LangGraph ingestion
    final_state = ingestion_flow.invoke(state)

    return final_state

def run_query(user_query: str):
    """
    Trigger full query retrieval pipeline.
    """
    state = QueryState(
        query=user_query,
        index_path="vector_store/faiss.index",
        metadata_path="vector_store/metadatas.pkl"
    )

    # Run LangGraph query pipeline
    final_state = query_flow.invoke(state)

    # Return final answer + highlight info + plots + dataframe (if applicable)
    return {
        "answer": final_state.get("answer"),
        "highlights": final_state.get("highlighted_pages"),
        "plot_data": final_state.get("generated_plots"),
        "dataframe": final_state.get("retrieved_dataframe"),
        'query_modality': final_state.get('query_modality')
    }