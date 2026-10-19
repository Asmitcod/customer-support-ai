from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama


MODEL_NAME = "llama3.1:latest"


def _call_ollama(system_prompt: str, user_prompt: str) -> str:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", user_prompt),
        ]
    )
    model = ChatOllama(model=MODEL_NAME, temperature=0.2)
    chain = prompt | model
    response = chain.invoke({})
    return response.content.strip()


import re


def _clean_response(text: str) -> str:
    text = text.strip()
    # Strip wrapping quotes if present
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        text = text[1:-1].strip()

    # Filter out supervisor meta-commentary like "Here is an improved version:" or "I made a few tweaks:"
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        lower = line.strip().lower()
        if re.match(r"^(here('s| is)|improved response|here is an improved|below is the revised).*", lower):
            continue
        if re.match(r"^(i made a few tweaks|i('ve| have) made|changes made|tweaks:|notes:|here are the tweaks).*", lower):
            break
        cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines).strip()
    return cleaned if cleaned else text


def generate_agent_response(intent: str, query: str, retrieved_context: str) -> str:
    system_prompt = (
        f"You are the {intent} customer support agent. "
        "Answer the customer's query directly using the provided company context. "
        "Be concise, helpful, and professional. Do not invent policy details. "
        "Output ONLY the final response to the customer. Do not add conversational fluff or meta-explanations."
    )
    user_prompt = f"Customer query:\n{query}\n\nRetrieved company context:\n{retrieved_context}\n\nResponse to customer:"

    try:
        raw = _call_ollama(system_prompt, user_prompt)
        return _clean_response(raw)
    except Exception as exc:
        return (
            f"{intent} response: I can help with this request. Based on the available "
            f"company information, here are the relevant details:\n\n{retrieved_context}\n\n"
            f"Local Ollama response generation was unavailable: {exc}"
        )


def supervisor_review(intent: str, query: str, draft_response: str) -> str:
    system_prompt = (
        "You are a customer support supervisor. "
        "Refine the draft response to be polite, concise, and clear. "
        "CRITICAL INSTRUCTIONS:\n"
        "1. Output ONLY the message that will be shown directly to the customer.\n"
        "2. Do NOT include phrases like 'Here is an improved version' or explanations of what tweaks you made.\n"
        "3. Do NOT include bullet points explaining your edits.\n"
        "4. Do NOT wrap the entire response in quotes."
    )
    user_prompt = (
        f"Intent: {intent}\n"
        f"Customer query: {query}\n\n"
        f"Draft response:\n{draft_response}\n\n"
        "Polished response to customer:"
    )

    try:
        raw = _call_ollama(system_prompt, user_prompt)
        return _clean_response(raw)
    except Exception:
        return _clean_response(draft_response)

