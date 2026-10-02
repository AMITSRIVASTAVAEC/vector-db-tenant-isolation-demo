#!/usr/bin/env python3
"""
Vector Database Tenant Isolation & ACL Bleed Demonstration
---------------------------------------------------------
Compares:
1. Naive Top-k Retrieval (ACL Bleed vulnerability)
2. Post-Retrieval Filtering (Context Starvation failure)
3. Metadata Pre-Filtering Envelope + Token-Level Post-Verification (Secure Enterprise RAG)
"""

import os
import sys
import time
from typing import Any, Dict, List
from pinecone import Pinecone, ServerlessSpec
from tabulate import tabulate

# ---------------------------------------------------------------------------
# 1. Embedding Model Setup
# ---------------------------------------------------------------------------
try:
    from sentence_transformers import SentenceTransformer
    print("[INIT] Loading local embedding model: all-MiniLM-L6-v2 (384 dims)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    DIMENSION = 384

    def embed_text(text: str) -> List[float]:
        return model.encode(text).tolist()

except ImportError:
    print("[WARN] sentence-transformers not found. Falling back to synthetic vector mock.")
    import random
    DIMENSION = 384

    def embed_text(text: str) -> List[float]:
        random.seed(abs(hash(text)) % (10**8))
        return [random.gauss(0, 1) for _ in range(DIMENSION)]


# ---------------------------------------------------------------------------
# 2. Pinecone Index Lifecycle
# ---------------------------------------------------------------------------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    print("[ERROR] PINECONE_API_KEY environment variable is missing.")
    print("Run: export PINECONE_API_KEY='your-key-here'")
    sys.exit(1)

pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "enterprise-acl-demo"

def initialize_index():
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    if INDEX_NAME in existing_indexes:
        print(f"[INDEX] Cleaning up existing index: {INDEX_NAME}...")
        pc.delete_index(INDEX_NAME)
        time.sleep(2)

    print(f"[INDEX] Provisioning serverless index: {INDEX_NAME}...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

    while not pc.describe_index(INDEX_NAME).status["ready"]:
        time.sleep(1)

    print(f"[INDEX] Index {INDEX_NAME} is online and ready.\n")
    return pc.Index(INDEX_NAME)


# ---------------------------------------------------------------------------
# 3. Document Ingestion with Relational Metadata & ACLs
# ---------------------------------------------------------------------------
DOCUMENTS: List[Dict[str, Any]] = [
    # --- Tenant Alpha (Acme Corp) ---
    {
        "id": "doc_eng_401",
        "text": "Q3 Engineering sprint velocity metrics, Jira roadmap milestones, and microservice refactoring targets.",
        "tenant_id": "tenant_alpha",
        "access_roles": ["engineering", "all_employees"],
        "classification": "internal",
    },
    {
        "id": "doc_eng_402",
        "text": "Quarterly leadership goals for engineering team: CI/CD automation, unit test coverage, and cloud budget management.",
        "tenant_id": "tenant_alpha",
        "access_roles": ["engineering", "all_employees"],
        "classification": "internal",
    },
    {
        "id": "doc_wiki_001",
        "text": "Acme Corp all-hands company handbook, standard remote work policies, and employee wellness benefits.",
        "tenant_id": "tenant_alpha",
        "access_roles": ["all_employees"],
        "classification": "public_internal",
    },
    {
        "id": "doc_hr_exec_909",
        "text": "Executive performance bonuses, VP equity grants, C-suite retention packages, and leadership compensation targets for Q4.",
        "tenant_id": "tenant_alpha",
        "access_roles": ["hr_executives", "c_suite"],
        "classification": "restricted_confidential",
    },
    {
        "id": "doc_board_910",
        "text": "Project Titan: Confidential M&A board deck, acquisition valuation multiple, and executive leadership succession planning.",
        "tenant_id": "tenant_alpha",
        "access_roles": ["c_suite", "board_members"],
        "classification": "restricted_confidential",
    },
    # --- Tenant Beta (Globex Inc) ---
    {
        "id": "doc_beta_101",
        "text": "Globex quarterly enterprise sales quotas and leadership revenue metrics for commercial accounts.",
        "tenant_id": "tenant_beta",
        "access_roles": ["sales", "all_employees"],
        "classification": "internal",
    },
]

def seed_database(index):
    print("[INGEST] Embedding and upserting document chunks with security metadata...")
    vectors = []
    for doc in DOCUMENTS:
        vec = {
            "id": doc["id"],
            "values": embed_text(doc["text"]),
            "metadata": {
                "text": doc["text"],
                "tenant_id": doc["tenant_id"],
                "access_roles": doc["access_roles"],
                "classification": doc["classification"],
            },
        }
        vectors.append(vec)

    index.upsert(vectors=vectors)
    # Allow indexing commit
    time.sleep(3)
    print(f"[INGEST] Successfully indexed {len(vectors)} chunks.\n")


# ---------------------------------------------------------------------------
# 4. Identity Context Definition (The Intern)
# ---------------------------------------------------------------------------
INTERN_SESSION = {
    "user_id": "usr_intern_4421",
    "email": "alex.intern@acme.com",
    "tenant_id": "tenant_alpha",
    "roles": ["engineering", "all_employees"],  # Lacks 'c_suite' and 'hr_executives'
}


# ---------------------------------------------------------------------------
# 5. Pipeline 1: Naive Top-k Retrieval (VULNERABLE: Causes ACL Bleed)
# ---------------------------------------------------------------------------
def naive_retrieval(index, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    query_vector = embed_text(query)
    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
    )
    return results.matches


# ---------------------------------------------------------------------------
# 6. Pipeline 2: Post-Filtering (BROKEN: Causes Context Starvation)
# ---------------------------------------------------------------------------
def post_filter_retrieval(index, query: str, user_session: Dict[str, Any], top_k: int = 3) -> List[Dict[str, Any]]:
    # Step 1: Fetch raw top_k from vector index without security filter
    query_vector = embed_text(query)
    raw_results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
    )

    # Step 2: Application backend attempts to drop chunks post-retrieval
    user_roles = set(user_session["roles"])
    user_tenant = user_session["tenant_id"]

    filtered_matches = []
    for match in raw_results.matches:
        chunk_tenant = match.metadata.get("tenant_id")
        chunk_roles = set(match.metadata.get("access_roles", []))

        # Check tenant isolation & role intersection
        if chunk_tenant == user_tenant and chunk_roles.intersection(user_roles):
            filtered_matches.append(match)

    return filtered_matches


# ---------------------------------------------------------------------------
# 7. Pipeline 3: Metadata Pre-Filtering Envelope + Token Post-Check (SECURE)
# ---------------------------------------------------------------------------
def secure_prefiltered_retrieval(
    index,
    query: str,
    user_session: Dict[str, Any],
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    query_vector = embed_text(query)

    # 1. Enforce Atomic Security Envelope at Vector Engine level
    security_filter = {
        "$and": [
            {"tenant_id": {"$eq": user_session["tenant_id"]}},
            {"access_roles": {"$in": user_session["roles"]}},
        ]
    }

    results = index.query(
        vector=query_vector,
        top_k=top_k,
        filter=security_filter,
        include_metadata=True,
    )

    # 2. Defense-in-Depth: Application-layer token-level post-assertion
    verified_matches = []
    user_roles = set(user_session["roles"])
    user_tenant = user_session["tenant_id"]

    for match in results.matches:
        chunk_tenant = match.metadata.get("tenant_id")
        chunk_roles = set(match.metadata.get("access_roles", []))

        if chunk_tenant == user_tenant and chunk_roles.intersection(user_roles):
            verified_matches.append(match)
        else:
            # Critical incident log: vector engine returned a chunk violating the envelope
            print(f"[SECURITY ALERT] Envelope breach detected on match {match.id}!")

    return verified_matches


# ---------------------------------------------------------------------------
# 8. Output Formatter
# ---------------------------------------------------------------------------
def print_results_table(matches: List[Any], title: str, user_session: Dict[str, Any]):
    print("=" * 90)
    print(f" {title.upper()}")
    print("=" * 90)

    user_roles = set(user_session["roles"])
    user_tenant = user_session["tenant_id"]

    rows = []
    for m in matches:
        chunk_id = m.id
        score = f"{m.score:.4f}"
        tenant = m.metadata.get("tenant_id", "N/A")
        roles = ", ".join(m.metadata.get("access_roles", []))
        classification = m.metadata.get("classification", "N/A")
        text_snippet = m.metadata.get("text", "")[:45] + "..."

        is_tenant_authorized = tenant == user_tenant
        is_role_authorized = bool(set(m.metadata.get("access_roles", [])).intersection(user_roles))

        if is_tenant_authorized and is_role_authorized:
            status = "AUTHORIZED"
        else:
            status = "LEAK / BLEED"

        rows.append([chunk_id, score, tenant, roles, classification, status, text_snippet])

    headers = ["Chunk ID", "Cosine", "Tenant", "Roles", "Class", "Audit Status", "Text Snippet"]
    print(tabulate(rows, headers=headers, tablefmt="github"))
    print("\n")


# ---------------------------------------------------------------------------
# Main Execution Flow
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    index = initialize_index()
    seed_database(index)

    # The Intern's Prompt: semantically close to executive compensation & leadership goals
    QUERY = "What are our leadership team's target priorities, bonus metrics, and budget allocations?"

    print(f"USER CONTEXT : {INTERN_SESSION['email']} ({INTERN_SESSION['user_id']})")
    print(f"USER TENANT  : {INTERN_SESSION['tenant_id']}")
    print(f"USER ROLES   : {INTERN_SESSION['roles']}")
    print(f"USER QUERY   : \"{QUERY}\"\n")

    # =======================================================================
    # TEST 1: Naive Top-k Retrieval (Shows ACL Bleed)
    # =======================================================================
    print("[RUNNING] Scenario 1: Naive Retrieval without metadata filters...")
    naive_matches = naive_retrieval(index, QUERY, top_k=3)
    print_results_table(naive_matches, "Scenario 1: Naive Retrieval (ACL Bleed Vulnerable)", INTERN_SESSION)

    # =======================================================================
    # TEST 2: Post-Filtering (Shows Context Starvation)
    # =======================================================================
    print("[RUNNING] Scenario 2: Post-Retrieval Filtering...")
    post_filter_matches = post_filter_retrieval(index, QUERY, INTERN_SESSION, top_k=3)
    print_results_table(post_filter_matches, "Scenario 2: Post-Filtering (Context Starvation)", INTERN_SESSION)
    print(f"-> Requested: k=3 chunks | Delivered to LLM: {len(post_filter_matches)} chunks (Starvation!)")

    # =======================================================================
    # TEST 3: Metadata Pre-Filtering Envelope (The Enterprise Fix)
    # =======================================================================
    print("\n[RUNNING] Scenario 3: Pre-Filtering Envelope + Token Verification...")
    secure_matches = secure_prefiltered_retrieval(index, QUERY, INTERN_SESSION, top_k=3)
    print_results_table(secure_matches, "Scenario 3: Secure Pre-Filtering Envelope (Enterprise Production)", INTERN_SESSION)
    print(f"-> Requested: k=3 chunks | Delivered to LLM: {len(secure_matches)} chunks (Clean, Complete Isolation)")

    # Cleanup prompt
    user_input = input("\nDo you want to delete the demo Pinecone index now? (y/N): ")
    if user_input.lower() == "y":
        print(f"Deleting {INDEX_NAME}...")
        pc.delete_index(INDEX_NAME)
        print("Done.")