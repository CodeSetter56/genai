# llm.py
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

def answer_query(results, query):
    llm = ChatGroq(model="llama-3.3-70b-versatile",temperature=0, max_tokens=1000)
    
    prompt = ChatPromptTemplate.from_template("""
        You are a helpful assistant. Answer ONLY using the context below.
        Do NOT perform calculations unless all required numbers are explicitly present in the context.
        If the answer is not clearly stated, say "I don't know based on the provided document."
        
        Context: {context}
        
        Question: {question}
        """)
    
    context = "\n\n".join([doc.page_content for doc, _ in results]) # combines the content of the retrieved chunks into a single context string
    chain = prompt | llm # creates a chain that first formats the prompt and then passes it to the LLM
    answer = chain.invoke({"context": context, "question": query})
    print("\nAnswer:", answer.content)
    
    







