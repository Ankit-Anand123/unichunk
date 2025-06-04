from state import IngestionState
from unstructured.partition.auto import partition

def load_document(state: IngestionState) -> IngestionState:
    """
    Load a document and update the state with its elements.
    
    Args:
        state (IngestionState): The current state of the ingestion process.
        
    Returns:
        IngestionState: Updated state with loaded document elements.
    """
    try:
        # Load the document using unstructured's partitioning
        elements = partition(
            filename=state.file_path,
            extract_images_in_pdf=False,
            infer_table_structure=True,
            extract_coordinates=True
        )
        state.elements = elements
        return state
    except Exception as e:
        state.error = str(e)
        raise e