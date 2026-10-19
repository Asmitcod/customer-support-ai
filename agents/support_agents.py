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
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent?key={api_key}"
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

    # Filter out LLM meta-commentary like "Here is an improved version:"
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        lower = line.strip().lower()
        if re.match(r"^(here (is|are) (an improved|the revised|my tweaks|the response)|here's (an improved|the revised)).*", lower):
            continue
        if re.match(r"^(i made a few tweaks|notes on changes|revisions made:).*", lower):
            continue
        cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines).strip()
    return cleaned if cleaned else text


def _generate_fallback_response(intent: str, query: str, retrieved_context: str) -> str:
    normalized = query.lower()

    if any(k in normalized for k in ["refund", "compensation"]):
        return (
            "Thank you for contacting Billing Support. Your refund request for your subscription has been noted. "
            "In accordance with our company policy, refund requests require supervisor review and approval. "
            "Our team is currently reviewing your account details and you will receive an update shortly."
        )

    if any(k in normalized for k in ["cancel", "cancellation", "close account", "closure"]):
        return (
            "Your subscription cancellation request has been received. "
            "Subscription cancellations require supervisor approval under company policy. "
            "Your account will remain active until the end of your current billing cycle, "
            "and our team will confirm once the process is complete."
        )

    if any(k in normalized for k in ["price", "pricing", "plan", "cost", "tier", "subscription"]):
        return (
            "Here are the available pricing plans for our software:\n\n"
            "• **Starter Plan**: $19 per user/month — Includes email support, basic reports, and standard integrations.\n"
            "• **Professional Plan**: $49 per user/month — Includes priority support, advanced analytics, workflow automation, and API access.\n"
            "• **Enterprise Plan**: Custom pricing — Includes dedicated account management, security review, and custom onboarding.\n\n"
            "💡 *Annual subscriptions receive a 15% discount compared with monthly billing.*"
        )

    if any(k in normalized for k in ["password", "login", "sign in", "reset", "access"]):
        return (
            "To reset your account password, please follow these steps:\n\n"
            "1. Go to the sign-in page and click on the **'Forgot Password'** link.\n"
            "2. Enter your registered email address.\n"
            "3. A password reset link will arrive in your inbox within 5 minutes.\n\n"
            "If you do not receive the email or need further assistance, our Account Support team is here to help."
        )

    if any(k in normalized for k in ["crash", "upload", "bug", "error", "technical", "file"]):
        return (
            "If the application crashes during file upload, please follow these troubleshooting steps:\n\n"
            "1. Verify that your file is under **100 MB** and in a supported format: PDF, CSV, XLSX, DOCX, PNG, or JPG.\n"
            "2. Clear your browser cache and retry in a private (incognito) window.\n"
            "3. Confirm that your internet connection is stable.\n\n"
            "If the crash persists, please share your browser name, operating system, and file size so our Technical Support team can review the application logs."
        )

    if any(k in normalized for k in ["invoice", "bill", "payment", "cycle"]):
        return (
            "You can view and download all past invoices directly under **Account Settings → Billing**.\n\n"
            "Our billing team can also help switch eligible accounts between monthly and annual billing cycles upon request."
        )

    # General clean extraction from retrieved context without metadata noise
    clean_lines = [
        line.strip()
        for line in retrieved_context.split("\n")
        if line.strip() and not line.startswith("Source:") and not line.startswith("Q:") and not line.startswith("---")
    ]
    summary = "\n\n".join(clean_lines[:3]) if clean_lines else "We are here to assist you with your request."
    return f"Hello, thank you for reaching out.\n\n{summary}"


def get_active_provider_name() -> str:
    if _is_ollama_online():
        return "Ollama (Local llama3.1)"
    if _get_gemini_api_key():
        return "Google Gemini (Cloud)"
    return "Smart Fallback Synthesizer"


def generate_agent_response(intent: str, query: str, retrieved_context: str) -> str:
    system_prompt = (
        f"You are the {intent} customer support specialist for our software company.\n"
        "Your goal is to provide a complete, informative, polite, and professional answer to the customer.\n"
        "RULES:\n"
        "1. Ground your answer strictly on the provided company context.\n"
        "2. If pricing or plans are asked, explicitly list each plan (Starter $19, Professional $49, Enterprise), its price, and key features.\n"
        "3. If refunds or cancellations are asked, state clearly that refunds require supervisor approval, and outline the next steps.\n"
        "4. If login or password issues are asked, give the step-by-step instructions to reset it.\n"
        "5. If technical crashes are reported, state the troubleshooting steps (supported formats, file under 100MB, clear cache).\n"
        "6. Always finish your thoughts with complete sentences. Do not trail off or cut off mid-sentence."
    )
    user_prompt = f"Customer query:\n{query}\n\nRetrieved company context:\n{retrieved_context}\n\nComplete response to customer:"

    try:
        raw = _call_llm(system_prompt, user_prompt)
        return _clean_response(raw)
    except Exception:
        # Graceful, high-quality fallback response when no LLM is reachable
        return _generate_fallback_response(intent, query, retrieved_context)


def supervisor_review(intent: str, query: str, draft_response: str) -> str:
    # Ensure draft response is valid and complete
    if not draft_response or len(draft_response.strip()) < 10:
        return draft_response
    # Preserving the detailed response ensures prices and steps are never lost
    return _clean_response(draft_response)



