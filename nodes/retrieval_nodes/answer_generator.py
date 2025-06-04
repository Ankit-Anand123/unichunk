from state import QueryState
import ollama

def answer_generator(state: QueryState) -> QueryState:
    try:
        # Build context from retrieved chunks
        context = ""
        for i, chunk in enumerate(state.retrieved_chunks):
            context += f"{chunk['text']}\n\n"

        # Build final prompt (your template)
        final_prompt = f"""
You are a helpful and knowledgeable AI assistant with expertise in analyzing documents and providing comprehensive answers.

Instructions:
1. For general questions (greetings, how are you, etc.), respond naturally and conversationally.
2. For document-related questions, use the provided context as your PRIMARY source, but you may also:
   - Add relevant background information to enhance understanding
   - Provide additional context that helps explain the document content
   - Offer related insights that complement the document information
   - Make connections between document content and broader knowledge
3. Always prioritize accuracy and clearly distinguish between document content and additional insights.
4. If the document doesn't contain relevant information for a specific question, say so clearly and provide what general knowledge you can.

Document Context:
{context}

Question: {state.query}

Answer:"""

        # Call Ollama model
        response = ollama.chat(
            model="llama3.1:8b",
            messages=[
                {"role": "user", "content": final_prompt}
            ]
        )

        answer = response['message']['content'].strip()

        return state.copy(update={
            "answer": answer
        })

    except Exception as e:
        state.error = str(e)
        raise e