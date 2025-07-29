#!/usr/bin/env python3

import uuid
import logging
from pathlib import Path
from llama_stack_client import LlamaStackClient
from strip_markdown import strip_markdown

# remove logging we otherwise get by default
logging.getLogger("httpx").setLevel(logging.WARNING)

# Configuration
# model_id = "meta-llama/Llama-3.1-8B-Instruct"
# model_id = "my-model3"
model_id = "llama-4-scout-17b-16e-w4a16"
SHOW_RAG_DOCUMENTS = False

# Initialize client
client = LlamaStackClient(
    base_url="http://10.1.2.128:8321",
    timeout=120.0,
)


def main():
    ########################
    # Create the agent
    system_prompt = open("prompt.txt").read()

    agentic_system_create_response = client.agents.create(
        agent_config={
            "model": model_id,
            "instructions": system_prompt,
            "toolgroups": [
                {
                    "name": "builtin::rag/knowledge_search",
                    "args": {"vector_db_ids": ["laptop-refresh-knowledge-base"]},
                },
                "mcp::asset_database",
                "mcp::servicenow",
            ],
            "tool_choice": "auto",
            "input_shields": [],
            "output_shields": [],
            "max_infer_iters": 10,
        }
    )
    agent_id = agentic_system_create_response.agent_id
    print(agent_id)

    # Create a session that will be used to ask the agent a sequence of questions
    session_create_response = client.agents.session.create(
        agent_id, session_name="agent1"
    )
    session_id = session_create_response.session_id

    #############################
    # ASK QUESTIONS

    questions = [
        "Can I replace my laptop, my employee id is 1234",
        "Yes",
        "3",
        "proceed",
    ]

    for j in range(1):
        print(
            f"Iteration {j} ------------------------------------------------------------"
        )

        for i, question in enumerate(questions):
            print("QUESTION: " + question)

            response_stream = client.agents.turn.create(
                agent_id=agent_id,
                session_id=session_id,
                stream=True,
                messages=[{"role": "user", "content": question}],
            )

            # Handle streaming response
            response = ""
            for chunk in response_stream:
                # print(chunk)
                if hasattr(chunk, "event") and hasattr(chunk.event, "payload"):
                    if chunk.event.payload.event_type == "turn_complete":
                        response = (
                            response + chunk.event.payload.turn.output_message.content
                        )
                    elif (
                        chunk.event.payload.event_type == "step_complete"
                        and chunk.event.payload.step_type == "tool_execution"
                        and SHOW_RAG_DOCUMENTS
                    ):
                        # Extract and print RAG document content in readable format
                        step_details = chunk.event.payload.step_details
                        if (
                            hasattr(step_details, "tool_responses")
                            and step_details.tool_responses
                        ):
                            print("\n" + "=" * 60)
                            print("RAG DOCUMENTS RETRIEVED")
                            print("=" * 60)

                            for tool_response in step_details.tool_responses:
                                if (
                                    hasattr(tool_response, "content")
                                    and tool_response.content
                                ):
                                    for item in tool_response.content:
                                        if (
                                            hasattr(item, "text")
                                            and "Result" in item.text
                                        ):
                                            # This is a result item, extract the content
                                            text = item.text
                                            if "Content:" in text:
                                                # Extract content after "Content:"
                                                content_start = text.find(
                                                    "Content:"
                                                ) + len("Content:")
                                                content_end = text.find("\nMetadata:")
                                                if content_end == -1:
                                                    content_end = len(text)

                                                content = text[
                                                    content_start:content_end
                                                ].strip()
                                                result_num = (
                                                    text.split("\n")[0]
                                                    if "\n" in text
                                                    else "Result"
                                                )

                                                print(f"\n--- {result_num} ---")
                                                print(content)
                                                print("-" * 40)
                            print("=" * 60)

            print("  RESPONSE:" + response)


if __name__ == "__main__":
    main()
