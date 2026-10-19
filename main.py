from pathlib import Path

from graph.workflow import build_customer_support_graph
from memory.sqlite_memory import SQLiteMemory
from rag.retriever import KnowledgeRetriever, ensure_sample_documents


BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
DATABASE_PATH = BASE_DIR / "database" / "memory.db"


DEMO_QUERIES = [
    "What are the pricing plans available for your software?",
    "I forgot my account password.",
    "My application crashes whenever I upload a file.",
    "I need a refund for my annual subscription.",
    "What was my previous support issue?",
]


def run_demo(app) -> None:
    for index, query in enumerate(DEMO_QUERIES, start=1):
        print("\n\n" + "#" * 72)
        print(f"DEMO QUERY {index}")
        print("#" * 72)
        app.invoke({"query": query})


def run_interactive(app) -> None:
    print("\n--- Interactive Live Mode ---")
    print("Type your customer query and press Enter.")
    print("Type 'exit' or 'quit' to return.\n")
    while True:
        try:
            query = input("Ask support query: ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit", "q"]:
                print("Exiting interactive mode.")
                break
            print("\n" + "=" * 72)
            app.invoke({"query": query})
            print("\n")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break


def main() -> None:
    ensure_sample_documents(DOCS_DIR)
    memory = SQLiteMemory(DATABASE_PATH)
    retriever = KnowledgeRetriever(DOCS_DIR)
    app = build_customer_support_graph(memory=memory, retriever=retriever)

    print("\nAI-Powered Customer Support Automation System")
    print("Using LangGraph, LangChain, Ollama, RAG, and SQLite memory.")
    print("Model: llama3.1:latest")
    print("-" * 72)
    print("Select Mode:")
    print("  [1] Run 5 Standard Demo Queries")
    print("  [2] Interactive Mode (Ask any custom query live)")
    print("-" * 72)

    choice = input("Enter choice [1 or 2, default is 1]: ").strip()
    if choice == "2":
        run_interactive(app)
    else:
        run_demo(app)


if __name__ == "__main__":
    main()

