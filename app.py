import os
import sqlite3
from pathlib import Path
import streamlit as st
import pandas as pd

from graph.workflow import build_customer_support_graph, _requires_approval
from memory.sqlite_memory import SQLiteMemory
from rag.retriever import KnowledgeRetriever, ensure_sample_documents

# Setup paths
BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
DATABASE_PATH = BASE_DIR / "database" / "memory.db"
DIAGRAM_PATH = BASE_DIR / "Workflow Diagram.png"

# Page Configuration
st.set_page_config(
    page_title="AI Customer Support Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        font-size: 0.8rem;
        font-weight: 600;
        border-radius: 9999px;
        margin-right: 0.5rem;
    }
    .badge-blue { background-color: #DBEAFE; color: #1E40AF; }
    .badge-green { background-color: #D1FAE5; color: #065F46; }
    .badge-purple { background-color: #EDE9FE; color: #5B21B6; }
    .badge-amber { background-color: #FEF3C7; color: #92400E; }
    .stChatMessage { border-radius: 0.75rem; }
</style>
""", unsafe_allow_html=True)


# Initialize components
@st.cache_resource
def get_graph():
    ensure_sample_documents(DOCS_DIR)
    mem = SQLiteMemory(DATABASE_PATH)
    ret = KnowledgeRetriever(DOCS_DIR)
    return mem, ret

memory, retriever = get_graph()


def get_recent_memory_records():
    if not DATABASE_PATH.exists():
        return []
    with sqlite3.connect(DATABASE_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, customer_name, customer_query, detected_department, assistant_response, timestamp
            FROM interactions
            ORDER BY id DESC LIMIT 10
            """
        )
        return cursor.fetchall()


# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712109.png", width=60)
    st.title("Control Panel")
    
    st.markdown("### ⚡ Quick Demo Queries")
    demo_queries = [
        ("💰 Pricing Plans", "What are the pricing plans available for your software?"),
        ("🔑 Password Reset", "I forgot my account password."),
        ("🐞 App Crash", "My application crashes whenever I upload a file."),
        ("💳 Refund Request", "I need a refund for my annual subscription."),
        ("🧠 Memory Recall", "What was my previous support issue?"),
    ]
    
    for label, query_text in demo_queries:
        if st.button(label, use_container_width=True):
            st.session_state["queued_query"] = query_text

    st.markdown("---")
    st.markdown("### 🛡️ Human-in-the-Loop")
    approval_preference = st.radio(
        "Supervisor Approval for Sensitive Queries (Refunds, etc.):",
        options=["Auto-Approve", "Auto-Reject", "Interactive Prompt"],
        index=0,
        help="Controls how high-risk requests are approved during the demo"
    )

    st.markdown("---")
    st.markdown("### 💾 SQLite Memory Database")
    records = get_recent_memory_records()
    if records:
        df = pd.DataFrame(
            records,
            columns=["ID", "User", "Query", "Dept", "Response", "Timestamp"]
        )[["Timestamp", "Dept", "Query"]]
        st.dataframe(df, use_container_width=True, height=180)
    else:
        st.caption("No interactions recorded in memory yet.")

    if st.button("🗑️ Clear Memory Database", use_container_width=True):
        if DATABASE_PATH.exists():
            with sqlite3.connect(DATABASE_PATH) as conn:
                conn.execute("DELETE FROM interactions")
                conn.commit()
            st.success("Memory cleared!")
            st.rerun()

    st.markdown("---")
    with st.expander("📊 View Architecture Diagram"):
        if DIAGRAM_PATH.exists():
            st.image(str(DIAGRAM_PATH), caption="LangGraph Workflow", use_container_width=True)
        else:
            st.info("Workflow Diagram.png not found")


# Main Area
st.markdown('<div class="main-header">🤖 AI Customer Support System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    '<span class="badge badge-blue">LangGraph Multi-Agent</span>'
    '<span class="badge badge-green">Local RAG</span>'
    '<span class="badge badge-purple">Human-in-the-Loop</span>'
    '<span class="badge badge-amber">SQLite Persistent Memory</span>'
    '</div>',
    unsafe_allow_html=True
)

# Chat history initialization
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "Hello! I am your AI-powered customer support assistant. How can I help you today?",
            "trace": None,
        }
    ]

# Display past messages
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Display trace if available
        if msg.get("trace"):
            t = msg["trace"]
            with st.expander("🔍 Multi-Agent Execution Trace", expanded=False):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🎯 Intent", t.get("intent", "Unknown"))
                with col2:
                    st.metric("🤖 Assigned Agent", f"{t.get('intent', 'Support')} Agent")
                with col3:
                    st.metric("🛡️ Approval Status", t.get("approval_status", "Not Required"))
                
                st.markdown("**📚 Retrieved Documents:**")
                sources = t.get("sources", [])
                if sources:
                    for s in sources:
                        st.markdown(f"- {s}")
                else:
                    st.caption("No specific policy documents retrieved.")
                
                st.markdown("**📝 Retrieved Context Preview:**")
                st.code(t.get("retrieved_context", "None")[:400] + ("..." if len(t.get("retrieved_context", "")) > 400 else ""), language="text")
                st.caption(f"💾 Committed to SQLite: {DATABASE_PATH.name}")


# Handle input (either from chat_input or sidebar quick button)
user_query = st.chat_input("Ask a customer support question...")

if "queued_query" in st.session_state:
    user_query = st.session_state.pop("queued_query")

if user_query:
    # Append user query to chat
    st.session_state["messages"].append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Determine approval handler based on sidebar preference
    def custom_approval(query_str: str) -> bool:
        if approval_preference == "Auto-Reject":
            return False
        return True  # Auto-Approve

    # Build graph with custom approval
    app = build_customer_support_graph(
        memory=memory,
        retriever=retriever,
        approval_handler=custom_approval
    )

    with st.chat_message("assistant"):
        with st.spinner("Processing through LangGraph multi-agent workflow..."):
            result = app.invoke({"query": user_query})
            final_ans = result.get("final_response", "Sorry, I could not process your query.")
            
            trace_info = {
                "intent": result.get("intent"),
                "approval_status": result.get("approval_status"),
                "sources": result.get("sources", []),
                "retrieved_context": result.get("retrieved_context", ""),
            }

            st.markdown(final_ans)

            # Execution trace accordion
            with st.expander("🔍 Multi-Agent Execution Trace", expanded=True):
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("🎯 Intent", result.get("intent", "Unknown"))
                with c2:
                    st.metric("🤖 Assigned Agent", f"{result.get('intent', 'Support')} Agent")
                with c3:
                    st.metric("🛡️ Approval Status", result.get("approval_status", "Not Required"))

                st.markdown("**📚 Retrieved Documents:**")
                sources = result.get("sources", [])
                if sources:
                    for s in sources:
                        st.markdown(f"- {s}")
                else:
                    st.caption("No specific policy documents retrieved.")

                st.markdown("**📝 Retrieved Context Preview:**")
                st.code(result.get("retrieved_context", "None")[:400] + ("..." if len(result.get("retrieved_context", "")) > 400 else ""), language="text")
                st.caption(f"💾 Committed to SQLite: {DATABASE_PATH.name}")

    # Save to history
    st.session_state["messages"].append({
        "role": "assistant",
        "content": final_ans,
        "trace": trace_info
    })
    st.rerun()
