import ollama
from state import QueryState

def query_modality_classifier(state: QueryState) -> QueryState:
    try:
        system_prompt = """
You are a query type classifier for a document assistant.

Your job is to analyze user queries and categorize them into one of 4 types:

1. text_only - for general questions, explanations, definitions, summaries, descriptions.
2. table_required - if the query asks for data, numbers, lists, tabular information.
3. visual_required - if the query asks for diagrams, images, figures, charts, graphs.
4. mixed - if query combines multiple types.

Output ONLY the type as one word (text_only, table_required, visual_required, mixed).

Some examples:
---
Q: What is medical cybersecurity?
A: text_only

Q: Can you show me the list of requirements?
A: table_required

Q: Explain the diagram about MDR Annex I.
A: visual_required

Q: What are the cybersecurity principles with supporting tables?
A: mixed
---
Now classify:
"""

        full_prompt = system_prompt + f"\nQ: {state.query}\nA:"

        response = ollama.chat(
            model="llama3.1:8b", 
            messages=[{"role": "user", "content": full_prompt}]
        )

        output_text = response["message"]["content"].strip().lower()

        if output_text not in ["text_only", "table_required", "visual_required", "mixed"]:
            output_text = "text_only"  # Fallback

        state.query_modality = output_text
        return state

    except Exception as e:
        state.error = str(e)
        raise e
