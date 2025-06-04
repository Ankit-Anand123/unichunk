from state import QueryState
import pandas as pd
import re

def pandas_node(state: QueryState) -> QueryState:
    try:
        tables = []

        for chunk in state.retrieved_chunks:
            text = chunk.get("text", "")

            # Very simple parsing logic:
            # Try to detect CSV-like tables (rows separated by \n, columns by commas)
            lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
            if not lines or len(lines) < 2:
                continue

            # Assume first row is header
            header = [col.strip() for col in re.split(r'[,\t|]', lines[0])]
            rows = []

            for line in lines[1:]:
                cols = [col.strip() for col in re.split(r'[,\t|]', line)]
                if len(cols) == len(header):
                    rows.append(cols)

            if rows:
                df = pd.DataFrame(rows, columns=header)
                tables.append(df)

        # If multiple tables found, you can combine them or keep them separate
        combined_df = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()

        return state.copy(update={
            "retrieved_dataframe": combined_df
        })

    except Exception as e:
        state.error = str(e)
        raise e