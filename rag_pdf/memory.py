CHAT_HISTORY_LIMIT = 7

def format_chat_history(chat_history):
    if not chat_history:
        return "No previous conversation."

    recent = chat_history[-CHAT_HISTORY_LIMIT:]
    lines = []
    for turn in recent:
        lines.append(f"Human: {turn['human']}")
        lines.append(f"Assistant: {turn['assistant']}")

    return "\n".join(lines)

def rewrite_query(query, chat_history, get_model):
    if not chat_history:
        return query

    recent = chat_history[-CHAT_HISTORY_LIMIT:]
    history_str = format_chat_history(recent)
    rewrite_prompt = f"""Given this conversation history: {history_str}
                    Rewrite this follow-up question as a standalone search query.
                    
                    Rules:
                    - Only resolve clear pronouns like "it", "that", "they" that directly reference something in history
                    - Do not infer topic or context from history if the query can stand on its own
                    - Do not add answers, facts, company names, roles or any other information from history unless explicitly referenced in the question
                    - Only add enough context to make it standalone, nothing more
                    - When in doubt, keep the query as close to the original wording as possible
                    
                    Question: "{query}"
                    
                    Return only the rewritten query, nothing else."""

    response = get_model().invoke(rewrite_prompt)
    return str(response.content)

def decompose_query(query, get_model):
    prompt = f"""Analyze this question and determine if it contains multiple distinct information requests.
            If it does, split it into separate standalone search queries.
            If it is a single request, return it as-is.
            
            Question: "{query}"
            
            Rules:
            - Only split if there are genuinely separate pieces of information being requested
            - Each sub-query should be fully standalone
            - Return one query per line, nothing else"""

    response = get_model().invoke(prompt)
    sub_queries = [q.strip() for q in str(response.content).strip().split("\n") if q.strip()]
    return sub_queries