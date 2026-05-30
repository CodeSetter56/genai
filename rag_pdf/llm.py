from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

def format_chat_history(chat_history):
    if not chat_history:
        return "No previous conversation."
    
    lines = []
    for turn in chat_history:
        lines.append(f"Human: {turn['human']}")
        lines.append(f"Assistant: {turn['assistant']}")
    
    return "\n".join(lines)

def answer_query(results, query, chat_history):
    llm = ChatGroq(model="llama-3.3-70b-versatile",temperature=0, max_tokens=1000)
    
    prompt = ChatPromptTemplate.from_template("""
        You are a helpful assistant. Answer ONLY using the context below.
        Do NOT perform calculations unless all required numbers are explicitly present in the context.
        If the answer is not clearly stated, say "I don't know based on the provided document."
        
        Previous conversation:
        {chat_history}
        
        Context: {context}
        
        Question: {question}
        """)
    
    context = "\n\n".join([doc.page_content for doc, _ in results]) # combines the content of the retrieved chunks into a single context string
    history_str = format_chat_history(chat_history)
    
    chain = prompt | llm # creates a chain that first formats the prompt and then passes it to the LLM
    
    print("\nAnswer: ", end="", flush=True)
    full_response = ""
    for chunk in chain.stream({"context": context, "question": query, "chat_history": history_str}):
        content = str(chunk.content)
        print(content, end="", flush=True)
        full_response += content
    print()
    
    return full_response
    
    







