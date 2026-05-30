from llm import get_model

def route_query(query, available_pdfs):
    router_prompt = f"""You are a document router. Given a user query, decide which documents are needed.
    
    Available documents: {available_pdfs}
    
    User query: "{query}"
    
    Rules:
    - Return ONLY the filenames that are strictly necessary
    - If the query mentions a specific company, only return that company's documents
    - Only return multiple documents if the query explicitly asks to compare or needs info from multiple sources
    - Return filenames exactly as listed, comma separated, nothing else"""
    
    response = get_model().invoke(router_prompt)
    
    selected = [f.strip() for f in str(response.content).split(",")]
    
    valid = [f for f in selected if f in available_pdfs]
    if not valid:
        return available_pdfs
    
    return valid