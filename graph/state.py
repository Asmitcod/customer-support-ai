from typing import TypedDict


class CustomerSupportState(TypedDict, total=False):
    customer_name: str
    query: str
    intent: str
    retrieved_context: str
    approval_required: bool
    approval_status: str
    previous_issue: str
    final_response: str
