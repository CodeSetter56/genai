from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# to prevent model creation before .env gets loded
_model = None
def get_model():
    global _model
    if _model is None:
        _model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, max_tokens=1000)
    return _model

def format_chat_history(chat_history):
    if not chat_history:
        return "No previous conversation."
    
    lines = []
    for turn in chat_history:
        lines.append(f"Human: {turn['human']}")
        lines.append(f"Assistant: {turn['assistant']}")
    
    return "\n".join(lines)

def rewrite_query(query, chat_history):
    if not chat_history:
        return query

    # format the chat history into a string that can be included in the prompt
    history_str = format_chat_history(chat_history)

    # rewrite the follow-up question as a standalone query using the conversation history
    rewrite_prompt = f"""Given this conversation history: {history_str}
    Rewrite this follow-up question as a standalone search query:"{query}"
    Return only the rewritten query, nothing else."""

    response = get_model().invoke(rewrite_prompt)
    return str(response.content)

def answer_query(results, query, chat_history):
    prompt = ChatPromptTemplate.from_template("""
        You are a helpful assistant.
        
        Use the provided context as your PRIMARY source of truth.
        You may supplement with your general knowledge ONLY when the document provides 
        partial information and general knowledge can reasonably fill the gap.
        When you use general knowledge, clearly mention it.
        Do NOT hallucinate specific numbers or facts not present in either the context or your training.
        If you truly cannot answer, say "I don't know based on the provided document."
        
        Previous conversation:
        {chat_history}
        
        Context: {context}
        
        Question: {question}
        """)
    
    context = "\n\n".join([doc.page_content for doc, _ in results]) # combines the content of the retrieved chunks into a single context string
    history_str = format_chat_history(chat_history)
    
    chain = prompt | get_model() # creates a chain that first formats the prompt and then passes it to the LLM
    
    print("\nAnswer: ", end="", flush=True)
    full_response = ""
    for chunk in chain.stream({"context": context, "question": query, "chat_history": history_str}):
        content = str(chunk.content)
        print(content, end="", flush=True)
        full_response += content
    print()
    
    return full_response