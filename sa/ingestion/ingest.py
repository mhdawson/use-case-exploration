#!/usr/bin/env python3

import uuid
import logging
from pathlib import Path
from llama_stack_client import LlamaStackClient

# remove logging we otherwise get by default
logging.getLogger("httpx").setLevel(logging.WARNING)

# Initialize client
client = LlamaStackClient(
    base_url="http://10.1.2.128:8321",
    timeout=120.0,
)


def main():
    ########################
    # Create the RAG database

    # use the first available provider
    providers = client.providers.list()
    provider = next(p for p in providers if p.api == "vector_io")

    # register the vector database
    vector_db_id = "laptop-refresh-knowledge-base"
    client.vector_dbs.register(
        vector_db_id=vector_db_id,
        provider_id=provider.provider_id,
        embedding_model="all-MiniLM-L6-v2",
    )

    # read in all of the files to be used with RAG
    rag_documents = []
    docs_path = Path("./docs")

    i = 0
    for file_path in docs_path.rglob("*.txt"):
        i += 1
        if file_path.is_file():
            with open(file_path, "r", encoding="utf-8") as f:
                plain_text = f.read()

            print(file_path)
            rag_documents.append(
                {
                    "document_id": f"doc-{i}",
                    "content": plain_text,
                    "mime_type": "text/plain",
                    "metadata": {},
                }
            )

    print("Inserting documents")
    client.tool_runtime.rag_tool.insert(
        documents=rag_documents,
        vector_db_id=vector_db_id,
        chunk_size_in_tokens=1000,
    )
    print("Finished inserting")


if __name__ == "__main__":
    main()
