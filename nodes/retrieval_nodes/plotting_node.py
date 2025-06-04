from state import QueryState
import matplotlib.pyplot as plt
import io
import base64

def plotting_node(state: QueryState) -> QueryState:
    try:
        # This is placeholder logic for now:
        # For real systems you'd parse table content or numeric patterns from chunks

        # We'll simply plot number of retrieved chunks as a dummy plot
        num_chunks = len(state.retrieved_chunks)

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.bar(["Retrieved Chunks"], [num_chunks])
        ax.set_title("Dummy Chunk Count Visualization")

        # Convert plot to base64 image string (so we can store easily in state)
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        encoded_img = base64.b64encode(buf.read()).decode("utf-8")

        return state.copy(update={
            "generated_plots": [encoded_img]
        })

    except Exception as e:
        state.error = str(e)
        raise e