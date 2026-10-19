import re
from pathlib import Path
from typing import Callable

from langgraph.graph import END, START, StateGraph

from agents.support_agents import generate_agent_response, supervisor_review
from graph.state import CustomerSupportState
from memory.sqlite_memory import SQLiteMemory
from rag.retriever import KnowledgeRetriever


HIGH_RISK_KEYWORDS = [
    "refund",
    "cancellation",
    "cancel subscription",
    "subscription cancellation",
    "compensation",
    "account closure",
    "close my account",
    "management escalation",
    "escalation to management",
    "manager",
]


def _print_step(title: str, value: str = "") -> None:
    print("\n" + "=" * 72)
    print(title)
    if value:
        print("-" * 72)
        print(value)


def _extract_customer_name(query: str) -> str:
    match = re.search(r"\bmy name is ([A-Z][a-zA-Z]+)\b", query, re.IGNORECASE)
    if match:
        return match.group(1).title()
    return "Unknown"


def _classify_intent(query: str) -> str:
    normalized = query.lower()

    if any(phrase in normalized for phrase in ["previous support issue", "previous issue", "last issue", "remember"]):
        return "Memory Recall"

    if any(word in normalized for word in ["price", "pricing", "plan", "demo", "sales", "enterprise"]):
        return "Sales"

    if any(word in normalized for word in ["crash", "error", "bug", "upload", "technical", "application"]):
        return "Technical"

    if any(word in normalized for word in ["invoice", "billing", "payment", "refund", "subscription", "charge"]):
        return "Billing"

    if any(word in normalized for word in ["password", "login", "sign in", "account", "closure", "profile"]):
        return "Account"

    return "Technical"


def _requires_approval(query: str) -> bool:
    normalized = query.lower()
    return any(keyword in normalized for keyword in HIGH_RISK_KEYWORDS)


def build_customer_support_graph(
    memory: SQLiteMemory,
    retriever: KnowledgeRetriever,
    approval_handler: Callable[[str], bool] | None = None,
) -> Callable[[CustomerSupportState], CustomerSupportState]:
    workflow = StateGraph(CustomerSupportState)

    def classify_intent_node(state: CustomerSupportState) -> CustomerSupportState:
        query = state["query"]
        intent = _classify_intent(query)
        customer_name = _extract_customer_name(query)

        _print_step("Customer Query", query)
        print("\n↓\n")
        _print_step("Intent Detected", intent)

        return {
            **state,
            "customer_name": customer_name,
            "intent": intent,
            "approval_required": _requires_approval(query),
            "approval_status": "Not Required",
        }

    def route_by_intent(state: CustomerSupportState) -> str:
        intent = state["intent"]
        _print_step("Routing Decision", f"Routing to {intent}")
        if intent == "Sales":
            return "sales_agent"
        if intent == "Technical":
            return "technical_agent"
        if intent == "Billing":
            return "billing_agent"
        if intent == "Account":
            return "account_agent"
        return "memory_recall"

    def make_department_agent(intent: str) -> Callable[[CustomerSupportState], CustomerSupportState]:
        def department_agent(state: CustomerSupportState) -> CustomerSupportState:
            _print_step("Current Agent", f"{intent} Agent")
            retrieved_context, sources = retriever.retrieve(state["query"])
            _print_step("Retrieved Documents", ", ".join(sources) if sources else "No matching documents found.")

            if not retrieved_context:
                retrieved_context = "No matching company documents were found for this query."

            draft_response = generate_agent_response(intent, state["query"], retrieved_context)
            return {
                **state,
                "intent": intent,
                "sources": sources,
                "retrieved_context": retrieved_context,
                "final_response": draft_response,
            }

        return department_agent

    def memory_recall_node(state: CustomerSupportState) -> CustomerSupportState:
        _print_step("Current Agent", "Memory Recall Agent")
        previous_issue = memory.get_previous_issue(state.get("customer_name", "Unknown"))
        _print_step("Memory Read", previous_issue or "No previous support issue found.")

        if previous_issue:
            response = f"I found this in SQLite memory: {previous_issue}"
        else:
            response = "I could not find a previous support issue in SQLite memory."

        return {
            **state,
            "intent": "Memory Recall",
            "previous_issue": previous_issue or "",
            "retrieved_context": "SQLite memory lookup",
            "sources": ["memory.db"],
            "final_response": response,
            "approval_required": False,
            "approval_status": "Not Required",
        }

    def approval_node(state: CustomerSupportState) -> CustomerSupportState:
        approval_required = state.get("approval_required", False)
        _print_step("Approval Required", "Yes" if approval_required else "No")

        if not approval_required:
            return {**state, "approval_status": "Not Required"}

        if approval_handler is not None:
            approved = approval_handler(state["query"])
        elif state.get("approved_by_user") is not None:
            approved = bool(state["approved_by_user"])
        else:
            answer = input("Approve request? (y/n): ").strip().lower()
            approved = answer == "y"

        status = "Approved" if approved else "Rejected"
        response = state["final_response"]

        if not approved:
            response = (
                "This request requires supervisor approval and was not approved, "
                "so I cannot process it automatically. A human supervisor should review it."
            )

        return {**state, "approval_status": status, "final_response": response}

    def route_after_approval(state: CustomerSupportState) -> str:
        if state.get("approval_status") == "Rejected":
            return "memory_write"
        return "supervisor"

    def supervisor_node(state: CustomerSupportState) -> CustomerSupportState:
        _print_step("Supervisor Review", "Reviewing and improving the response.")
        reviewed_response = supervisor_review(
            state["intent"],
            state["query"],
            state["final_response"],
        )
        return {**state, "final_response": reviewed_response}

    def memory_write_node(state: CustomerSupportState) -> CustomerSupportState:
        memory.save_interaction(
            customer_name=state.get("customer_name", "Unknown"),
            customer_query=state["query"],
            detected_department=state["intent"],
            assistant_response=state["final_response"],
        )
        _print_step("Memory Write", f"Saved interaction to {Path(memory.database_path).name}.")
        _print_step("Final Response", state["final_response"])
        return state

    workflow.add_node("classify_intent", classify_intent_node)
    workflow.add_node("sales_agent", make_department_agent("Sales"))
    workflow.add_node("technical_agent", make_department_agent("Technical"))
    workflow.add_node("billing_agent", make_department_agent("Billing"))
    workflow.add_node("account_agent", make_department_agent("Account"))
    workflow.add_node("memory_recall", memory_recall_node)
    workflow.add_node("approval", approval_node)
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("memory_write", memory_write_node)

    workflow.add_edge(START, "classify_intent")
    workflow.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {
            "sales_agent": "sales_agent",
            "technical_agent": "technical_agent",
            "billing_agent": "billing_agent",
            "account_agent": "account_agent",
            "memory_recall": "memory_recall",
        },
    )

    for node_name in ["sales_agent", "technical_agent", "billing_agent", "account_agent"]:
        workflow.add_edge(node_name, "approval")

    workflow.add_edge("memory_recall", "supervisor")
    workflow.add_conditional_edges(
        "approval",
        route_after_approval,
        {
            "supervisor": "supervisor",
            "memory_write": "memory_write",
        },
    )
    workflow.add_edge("supervisor", "memory_write")
    workflow.add_edge("memory_write", END)

    return workflow.compile()
