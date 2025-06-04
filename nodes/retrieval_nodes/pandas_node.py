from state import QueryState
import pandas as pd
import ollama
import json

def pandas_node(state: QueryState) -> QueryState:
    try:
        # Build context from retrieved chunks
        context = ""
        for i, chunk in enumerate(state.retrieved_chunks):
            context += f"{chunk['text']}\n\n"

        # Build final prompt for structured table extraction
        final_prompt = f"""
You are an intelligent data extraction AI. 

Your task:
- Analyze the provided document context and the user query.
- If any tabular information exists related to the user query, extract it as JSON.
- If no table is found or applicable, simply return: "NO_TABLE_FOUND"

Rules:
- Extract only the rows and columns relevant to the query.
- Don't give any description about the table, just give only the rows and columns.
- Return JSON in the following format:
{{
  "columns": ["col1", "col2", "col3"],
  "rows": [
    ["row1_value1", "row1_value2", "row1_value3"],
    ["row2_value1", "row2_value2", "row2_value3"]
  ]
}}

Document Context:
{context}

User Query:
{state.query}

Your Response:
"""

        # Call Ollama model
        response = ollama.chat(
            model="llama3.1:8b",
            messages=[
                {"role": "user", "content": final_prompt}
            ]
        )

        answer = response['message']['content'].strip()

        # Handle response
        if "NO_TABLE_FOUND" in answer:
            extracted_df = pd.DataFrame()  # Empty dataframe
        else:
            # Extract JSON safely
            try:
                json_str = extract_json(answer)
                data = json.loads(json_str)
                extracted_df = pd.DataFrame(data["rows"], columns=data["columns"])
            except Exception as parse_err:
                state.error = f"Failed to parse JSON: {parse_err}"
                raise parse_err
            
        print(extracted_df)
        return state.copy(update={
            "retrieved_dataframe": extracted_df
        })

    except Exception as e:
        state.error = str(e)
        raise e


# Helper function to robustly extract JSON from LLM output
import re
def extract_json(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)
    else:
        raise ValueError("No JSON found in LLM response")
