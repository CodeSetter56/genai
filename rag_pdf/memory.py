def format_chat_history(chat_history):
    if not chat_history:
        return "No previous conversation."
    
    lines = []
    for turn in chat_history:
        lines.append(f"Human: {turn['human']}")
        lines.append(f"Assistant: {turn['assistant']}")
    
    return "\n".join(lines)

def rewrite_query(query, chat_history, get_model):
    if not chat_history:
        return query

    history_str = format_chat_history(chat_history)
    rewrite_prompt = f"""Given this conversation history: {history_str}
    Rewrite this follow-up question as a standalone search query:"{query}"
    Return only the rewritten query, nothing else."""

    response = get_model().invoke(rewrite_prompt)
    return str(response.content)