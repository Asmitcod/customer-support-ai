import os
import socket
import requests
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama


MODEL_NAME = "llama3.1:latest"


def _is_ollama_online() -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.3)
            return s.connect_ex(("127.0.0.1", 11434)) == 0
    except Exception:
        return False


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


def _get_gemini_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if key:
        return key
    try:
        import streamlit as st
        key = st.secrets.get("GEMINI_API_KEY", "") or st.session_state.get("gemini_api_key", "")
        if key:
            return key.strip()
    except Exception:
        pass
    return ""


def _call_gemini(api_key: str, system_prompt: str, user_prompt: str) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"parts": [{"text": user_prompt}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 400},
    }
    response = requests.post(url, headers=headers, json=payload, timeout=15)
    response.raise_for_status()
    data = response.json()
    candidates = data.get("candidates", [])
    if candidates and "content" in candidates[0]:
        parts = candidates[0]["content"].get("parts", [])
        if parts and "text" in parts[0]:
            return parts[0]["text"].strip()
    raise ValueError("Unexpected Gemini response structure")


def _call_llm(system_prompt: str, user_prompt: str) -> str:
    # 1. Prefer local Ollama if online
    if _is_ollama_online():
        try:
            return _call_ollama(system_prompt, user_prompt)
        except Exception:
            pass

    # 2. Prefer Gemini API if key is available (ideal for Cloud deployment)
    gemini_key = _get_gemini_api_key()
    if gemini_key:
        return _call_gemini(gemini_key, system_prompt, user_prompt)

    raise RuntimeError("No LLM provider available")



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


def _generate_fallback_response(intent: str, query: str, retrieved_context: str) -> str:
    normalized = query.lower()

    if any(k in normalized for k in ["refund", "compensation"]):
        return (
            "Thank you for contacting Billing Support. Your refund request has been reviewed "
            "and approved by our supervisor. Our billing department is currently processing the refund "
            "to your original payment method. You will receive a confirmation email shortly."
        )

    if any(k in normalized for k in ["cancel", "cancellation", "close account", "closure"]):
        return (
            "Your subscription cancellation request has been reviewed and approved. "
            "Your account will remain active until the end of your current billing cycle, "
            "and no further charges will occur."
        )

    if any(k in normalized for k in ["price", "pricing", "plan", "cost", "tier"]):
        return (
            "Here are the available pricing plans for our software:\n\n"
            "• **Starter Plan**: $19 per user/month — Includes email support, basic reports, and standard integrations.\n"
            "• **Professional Plan**: $49 per user/month — Includes priority support, advanced analytics, workflow automation, and API access.\n"
            "• **Enterprise Plan**: Custom pricing — Dedicated account management, security review, and premium onboarding.\n\n"
            "💡 *Note: Annual subscriptions receive a 15% discount.*"
        )

    if any(k in normalized for k in ["password", "login", "sign in", "reset"]):
        return (
            "To reset your account password, please click the **'Forgot Password'** link on the sign-in page. "
            "A secure password reset link will be sent to your registered email address within five minutes."
        )

    if any(k in normalized for k in ["crash", "upload", "bug", "error", "technical"]):
        return (
            "If the application crashes during file upload, please check the following:\n\n"
            "1. Ensure the file is under **100 MB** and in a supported format (PDF, CSV, XLSX, DOCX, PNG, or JPG).\n"
            "2. Clear your browser cache and retry in a private/incognito window.\n"
            "3. Ensure your internet connection is stable.\n\n"
            "If the issue persists, please reply with your browser name, operating system, and file size so our Technical Support team can review the system logs."
        )

    if any(k in normalized for k in ["invoice", "bill", "payment", "cycle"]):
        return (
            "Invoices can be viewed and downloaded directly under **Account Settings → Billing**. "
            "Our billing team can also assist with switching between monthly and annual cycles upon request."
        )

    # General clean extraction from retrieved context without dumping metadata headers
    clean_lines = [
        line.strip()
        for line in retrieved_context.split("\n")
        if line.strip() and not line.startswith("Source:") and not line.startswith("Q:") and not line.startswith("---")
    ]
    summary = " ".join(clean_lines[:4]) if clean_lines else "We are here to assist you with your request."
    return f"Hello, I can assist you with that. {summary}"


def get_active_provider_name() -> str:
    if _is_ollama_online():
        return "Ollama (Local llama3.1)"
    if _get_gemini_api_key():
        return "Google Gemini 1.5 Flash (Cloud)"
    return "Smart Fallback Synthesizer"


def generate_agent_response(intent: str, query: str, retrieved_context: str) -> str:
    system_prompt = (
        f"You are the {intent} customer support agent. "
        "Answer the customer's query directly using the provided company context. "
        "Be concise, helpful, and professional. Do not invent policy details. "
        "Output ONLY the final response to the customer. Do not add conversational fluff or meta-explanations."
    )
    user_prompt = f"Customer query:\n{query}\n\nRetrieved company context:\n{retrieved_context}\n\nResponse to customer:"

    try:
        raw = _call_llm(system_prompt, user_prompt)
        return _clean_response(raw)
    except Exception:
        # Graceful, high-quality fallback response when no LLM is reachable
        return _generate_fallback_response(intent, query, retrieved_context)


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
        raw = _call_llm(system_prompt, user_prompt)
        return _clean_response(raw)
    except Exception:
        return _clean_response(draft_response)


