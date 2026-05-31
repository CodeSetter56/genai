from llm import get_model

def generate_pdf_description(filename, chunks):
    sample_text = "\n".join([c.page_content for c in chunks[:3]])
    prompt = f"""In one sentence, describe what this document is about based on this excerpt: {sample_text}
            Return only the description, nothing else."""
    
    response = get_model().invoke(prompt)
    return str(response.content)

def route_query(query, pdf_descriptions):
    doc_list = "\n".join([f"- {name}: {desc}" for name, desc in pdf_descriptions.items()])
    
    router_prompt = f"""You are a document router. Given a user query, decide which documents are needed.
    Available documents: {doc_list}
    User query: "{query}"
    
    Rules:
    - Return ONLY the filenames that are strictly necessary
    - If the query mentions a specific company, only return that company's documents  
    - Only return multiple documents if the query explicitly needs info from multiple sources
    - Return filenames exactly as listed, comma separated, nothing else"""
    
    response = get_model().invoke(router_prompt)
    selected = [f.strip() for f in str(response.content).split(",")]
    
    available = list(pdf_descriptions.keys())
    valid = [f for f in selected if f in available]
    
    if not valid:
        return available 
    
    return valid