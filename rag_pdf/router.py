from llm import get_model

def generate_pdf_description(filename, chunks):
    sample_text = "\n".join([c.page_content for c in chunks[:3]])
    prompt = f"""Analyze this document excerpt and return a structured description in exactly this format:
Type: <what kind of document this is>
Topic: <main subject or domain of the document>
Entities: <key people, numbers, names, locations, descriptions, companies, products mentioned, or N/A>
Contains: <comma separated list of specific information this document has>

Document excerpt:
{sample_text}

Return only the structured description, nothing else."""

    response = get_model().invoke(prompt)
    return str(response.content)

def route_query(query, pdf_descriptions):
    doc_list = "\n".join([f"- {name}:\n  {desc}" for name, desc in pdf_descriptions.items()])

    router_prompt = f"""You are a document router. Given a user query, decide which documents are needed.

Available documents:
{doc_list}

User query: "{query}"

Rules:
- Return ONLY the filenames that are strictly necessary
- Use the 'Contains' field to judge relevance — only select documents that contain information directly relevant to the query
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