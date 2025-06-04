from state import QueryState
import ollama
from tools import extract_json


def query_classifier(state: QueryState) -> QueryState:
    try:
        system_prompt = """
You are a query classifier for a document assistant.

Classify the user's query intent into two flags:
- plot_requested: true or false
- table_requested: true or false

Rules:
- If the user is simply asking whether tables/images exist (e.g. "Are there any tables?"), both flags must be false.
- If the user asks to extract tables or show tables as dataframe, set table_requested = true.
- If the user asks to generate plots or visualize data, set plot_requested = true.
- Never classify plot_requested = true when summary of the image is asked. It should alwasy be false

Return your output as pure JSON with two keys: plot_requested and table_requested.

Examples:

User query: "Generate a pie chart of revenue"
Output: {"plot_requested": true, "table_requested": false}

User query: "Extract tables from this document"
Output: {"plot_requested": false, "table_requested": true}

User query: "Are there any tables in this document?"
Output: {"plot_requested": false, "table_requested": false}
"""

        response = ollama.chat(
            model="llama3.1:8b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"User query: {state.query}"}
            ]
        )

        output_text = response['message']['content'].strip()
        parsed_output = extract_json(output_text)

        print(parsed_output)
        return state.copy(update={
            "plot_requested": parsed_output.get("plot_requested", False),
            "table_requested": parsed_output.get("table_requested", False),
        })

    except Exception as e:
        state.error = str(e)
        raise e