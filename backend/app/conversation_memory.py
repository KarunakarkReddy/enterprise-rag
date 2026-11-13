
# Stores conversation history separately for each session.

conversation_store = {}

MAX_HISTORY = 20


def get_history(session_id):
    return conversation_store.get(session_id, [])


def save_message(session_id, question, answer):
    if session_id not in conversation_store:
        conversation_store[session_id] = []

    conversation_store[session_id].append({
        "role": "user",
        "content": question
    })

    conversation_store[session_id].append({
        "role": "assistant",
        "content": answer
    })

    # Keep only the latest 20 messages
    conversation_store[session_id] = (
        conversation_store[session_id][-MAX_HISTORY:]
    )


def clear_history(session_id):
    conversation_store.pop(session_id, None)
