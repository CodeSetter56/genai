from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from memory import format_chat_history

# lazy initialization of the model to avoid dotenv mounting issues
_model = None
def get_model():
    global _model
    if _model is None:
        _model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, max_tokens=1000)
    return _model

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
    
    context = "\n\n".join([doc.page_content for doc, _ in results])
    history_str = format_chat_history(chat_history)

    chain = prompt | get_model() # prompt -> model -> response
    
    print("\nAnswer: ", end="", flush=True)
    full_response = ""
    for chunk in chain.stream({"context": context, "question": query, "chat_history": history_str}):
        content = str(chunk.content)
        print(content, end="", flush=True)
        full_response += content
    print()
    
    return full_response