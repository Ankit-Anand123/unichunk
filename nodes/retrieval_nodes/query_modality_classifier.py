import ollama
from state import QueryState
 
def query_modality_classifier(state: QueryState) -> QueryState:
    try:
        system_prompt = """
You are a query type classifier for a document assistant.
 
Your job is to analyze user queries and categorize them into one of 4 types:
 
1. **text_only** - for general questions, explanations, definitions, summaries, descriptions, factual questions about when/what/how/why something happened or exists
2. **table_required** - ONLY if the user explicitly asks to extract, show, display, or analyze tabular data/tables/lists as data structures
3. **visual_required** - ONLY if the user explicitly asks about diagrams, images, figures, charts, graphs, or visual content
4. **mixed** - if query combines multiple types explicitly
 
IMPORTANT RULES:
- Questions about dates, regulations, numbers, or factual information are **text_only**
- Only classify as "table_required" if user wants to SEE or EXTRACT table data
- Only classify as "visual_required" if user wants to SEE or ANALYZE visual content
- Don't confuse factual questions with table/visual requests
 
Output ONLY the type as one word (text_only, table_required, visual_required, mixed).
 
Examples:
---
Q: When were the regulations 745/2017 adopted?
A: text_only
 
Q: What are the cybersecurity requirements?
A: text_only
 
Q: Show me the tables from this document
A: table_required
 
Q: Extract the data table about requirements
A: table_required
 
Q: Are there any tables in this document?
A: text_only
 
Q: Explain the diagram about security
A: visual_required
 
Q: Show me the flowchart
A: visual_required
 
Q: What does the image show?
A: visual_required
 
Q: What is the definition of cybersecurity with supporting tables?
A: mixed
 
Q: Give me both the explanation and extract the related tables
A: mixed
---
Now classify:
"""
 
        full_prompt = system_prompt + f"\nQ: {state.query}\nA:"
 
        response = ollama.chat(
            model="phi3:3.8b",  # Using the same model as query_classifier for consistency
            messages=[{"role": "user", "content": full_prompt}]
        )
 
        output_text = response["message"]["content"].strip().lower()
 
        # Clean up the output - sometimes models add extra text
        if "text_only" in output_text:
            output_text = "text_only"
        elif "table_required" in output_text:
            output_text = "table_required"
        elif "visual_required" in output_text:
            output_text = "visual_required"
        elif "mixed" in output_text:
            output_text = "mixed"
        else:
            # Default fallback for unclear responses
            output_text = "text_only"
 
        print(f"Query: {state.query}")
        print(f"Classified modality: {output_text}")
 
        state.query_modality = output_text
        return state
 
    except Exception as e:
        # If classification fails, default to text_only
        print(f"Error in modality classification: {e}")
        state.query_modality = "text_only"
        state.error = str(e)
        return state  # Don't raise, just continue with default