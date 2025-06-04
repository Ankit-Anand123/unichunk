from state import IngestionState

def content_classifier(state: IngestionState) -> IngestionState:
    """
    Analyze elements to determine available content types (text, table, visual).
    """
    print('inside content classifier node')
    try:
        elements = state.elements
        state.has_text = any(
            el.category.lower() in ["title", "narrativetext", "listitem", "figurecaption"] for el in elements
        )
        state.has_table = any(
            el.category.lower() == "table" for el in elements
        )

        state.has_visual = any(
            el.category.lower() == "image" for el in elements
        )

        print(state.has_table, state.has_text, state.has_visual)
        return state
    except Exception as e:
        state.error = str(e)
        raise e
