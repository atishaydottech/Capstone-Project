"""OpenAI LLM generation for RAG answers."""

import os

DEFAULT_MODEL = "gpt-5.4-mini"
FAST_MODEL = "gpt-5.4-nano"


def _get_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")
    import openai
    return openai.OpenAI(api_key=api_key)


def complete(user_msg, model=FAST_MODEL, temperature=0, max_tokens=1024):
    """Single-turn OpenAI completion. Returns text or raises."""
    client = _get_client()
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": user_msg}],
        temperature=temperature,
    )
    return response.choices[0].message.content


def generate_answer(system_msg, user_msg, model=DEFAULT_MODEL):
    """
    Generate an answer using OpenAI.

    Returns (answer_text, error_message). Exactly one will be None.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None, "Set `OPENAI_API_KEY` in `.env` to enable LLM generation."

    try:
        import openai
    except ImportError:
        return None, "Install `openai` package to enable LLM generation."

    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            max_tokens=1024,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0,
        )
        answer = response.choices[0].message.content
        return answer, None
    except Exception as e:
        return None, f"LLM error: {e}"
