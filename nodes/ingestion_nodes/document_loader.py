from state import IngestionState
from unstructured.partition.pdf import partition_pdf

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
        elements = partition_pdf(
            filename=state.file_path,
            strategy="hi_res",
            extract_image_block_types=["Image"],
            extract_image_block_to_payload=True
        )
        state.elements = elements
        return state
    except Exception as e:
        state.error = str(e)
        raise e