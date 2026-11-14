
# --------------------------------------------------
# CONVERSATION MEMORY CONFIGURATION
# --------------------------------------------------

conversation_store = {}

MAX_HISTORY = 20


# --------------------------------------------------
# VALIDATE SESSION ID
# --------------------------------------------------

def validate_session_id(session_id):
    """
    Validate the conversation session ID.
    """

    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("Session ID cannot be empty.")

    if len(session_id) > 100:
        raise ValueError("Session ID is too long.")

    return session_id.strip()


# --------------------------------------------------
# CREATE A SESSION KEY
# --------------------------------------------------

def get_session_key(session_id, document_id=None):
    """
    Keep conversation history separate for each
    session and selected document.
    """

    session_id = validate_session_id(session_id)

    if document_id is not None:
        try:
            document_id = int(document_id)
        except (TypeError, ValueError):
            raise ValueError("Invalid document ID.")

        if document_id <= 0:
            raise ValueError("Document ID must be positive.")

    document_key = (
        str(document_id)
        if document_id is not None
        else "all_documents"
    )

    return f"{session_id}:{document_key}"


# --------------------------------------------------
# GET CONVERSATION HISTORY
# --------------------------------------------------

def get_history(session_id, document_id=None):
    """
    Retrieve previous messages for the session.
    """

    session_key = get_session_key(
        session_id,
        document_id
    )

    return conversation_store.get(
        session_key, []
    ).copy()


# --------------------------------------------------
# SAVE CONVERSATION
# --------------------------------------------------

def save_message(
    session_id,
    question,
    answer,
    document_id=None,
    evidence=None
):
    """
    Save the user question and assistant answer.
    """

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty.")

    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("Answer cannot be empty.")

    if evidence is not None and not isinstance(evidence, str):
        raise ValueError("Evidence must be text.")

    session_key = get_session_key(
        session_id,
        document_id
    )

    if session_key not in conversation_store:
        conversation_store[session_key] = []

    assistant_message = {
        "role": "assistant",
        "content": answer.strip()
    }

    if evidence and evidence.strip():
        assistant_message["evidence"] = evidence.strip()

    conversation_store[session_key].extend([
        {
            "role": "user",
            "content": question.strip()
        },
        assistant_message
    ])

    # Keep only the latest 20 messages.
    conversation_store[session_key] = (
        conversation_store[session_key][-MAX_HISTORY:]
    )


# --------------------------------------------------
# CLEAR CONVERSATION HISTORY
# --------------------------------------------------

def clear_history(session_id, document_id=None):
    """
    Clear conversation history for a session.
    """

    session_key = get_session_key(
        session_id,
        document_id
    )

    conversation_store.pop(session_key, None)
