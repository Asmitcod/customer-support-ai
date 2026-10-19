import math
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, List

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


DEFAULT_DOCS = {
    "company_policy.txt": """Company Policy

Customers may contact support for sales, technical, billing, and account help.
Refunds are reviewed by Billing and require human supervisor approval.
Subscription cancellations, account closures, compensation requests, and management escalations require human approval.
Support agents should be polite, concise, and transparent about next steps.
""",
    "pricing_guide.txt": """Pricing Guide

Starter Plan: $19 per user per month. Includes email support, basic reports, and standard integrations.
Professional Plan: $49 per user per month. Includes priority support, advanced analytics, workflow automation, and API access.
Enterprise Plan: Custom pricing. Includes dedicated account management, security review, custom onboarding, and premium support.
""",
    "technical_manual.txt": """Technical Manual

If the application crashes during file upload, first verify that the file is under 100 MB and uses a supported format.
Ask the customer to clear the browser cache, retry in a private browser window, and confirm that their internet connection is stable.
If crashes continue, collect the browser name, operating system, file type, file size, and approximate time of the failed upload.
""",
    "faq.txt": """FAQ

Q: How do I reset my password?
A: Use the Forgot Password link on the sign-in page. A reset email should arrive within five minutes.

Q: Where can I see invoices?
A: Invoices are available under Account Settings, then Billing.
""",
}


def ensure_sample_documents(docs_dir: Path) -> None:
    docs_dir.mkdir(parents=True, exist_ok=True)
    for file_name, contents in DEFAULT_DOCS.items():
        path = docs_dir / file_name
        if not path.exists():
            path.write_text(contents, encoding="utf-8")


class LocalKeywordVectorStore:
    """Small local vector store using bag-of-words cosine similarity."""

    def __init__(self, documents: Iterable[Document]) -> None:
        self.documents = list(documents)
        self.vectors = [self._vectorize(document.page_content) for document in self.documents]

    @staticmethod
    def _tokens(text: str) -> List[str]:
        return re.findall(r"[a-zA-Z][a-zA-Z0-9]+", text.lower())

    def _vectorize(self, text: str) -> Counter:
        return Counter(self._tokens(text))

    @staticmethod
    def _cosine(left: Counter, right: Counter) -> float:
        common_terms = set(left) & set(right)
        dot_product = sum(left[term] * right[term] for term in common_terms)
        left_norm = math.sqrt(sum(value * value for value in left.values()))
        right_norm = math.sqrt(sum(value * value for value in right.values()))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return dot_product / (left_norm * right_norm)

    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        query_vector = self._vectorize(query)
        scored_documents = [
            (self._cosine(query_vector, vector), document)
            for vector, document in zip(self.vectors, self.documents)
        ]
        scored_documents.sort(key=lambda item: item[0], reverse=True)
        return [document for score, document in scored_documents[:k] if score > 0]


class KnowledgeRetriever:
    def __init__(self, docs_dir: Path) -> None:
        ensure_sample_documents(docs_dir)
        documents = self._load_documents(docs_dir)
        splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=80)
        chunks = splitter.split_documents(documents)
        self.vector_store = LocalKeywordVectorStore(chunks)

    def _load_documents(self, docs_dir: Path) -> List[Document]:
        documents: List[Document] = []
        for path in sorted(docs_dir.glob("*.txt")):
            loader = TextLoader(str(path), encoding="utf-8")
            documents.extend(loader.load())
        return documents

    def retrieve(self, query: str, k: int = 3) -> tuple[str, list[str]]:
        documents = self.vector_store.similarity_search(query, k=k)
        context_blocks = []
        sources = []

        for document in documents:
            source = Path(str(document.metadata.get("source", "unknown"))).name
            sources.append(source)
            context_blocks.append(f"Source: {source}\n{document.page_content}")

        return "\n\n".join(context_blocks), sources
