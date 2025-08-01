#!/usr/bin/env python3

import uuid
import logging
import random
from pathlib import Path
from llama_stack_client import LlamaStackClient
from strip_markdown import strip_markdown

# remove logging we otherwise get by default
logging.getLogger("httpx").setLevel(logging.WARNING)

# Configuration
model_id = "meta-llama/Llama-3.1-8B-Instruct"
#model_id = "llama-4-scout-17b-16e-w4a16"

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
            "tool_choice": "auto",
            "input_shields": [],
            "output_shields": [],
            "max_infer_iters": 10,
        }
    )
    agent_id = agentic_system_create_response.agent_id
    print(agent_id)

    #############################
    # ASK QUESTIONS

    questions = [
        # REFRESH_AGENT examples - laptop refresh/replacement
        {
            "question": "Can I replace my laptop, my employee id is 1234",
            "expected_response": "REFRESH_AGENT"
        },
        {
            "question": "What is the laptop refresh processs?",
            "expected_response": "REFRESH_AGENT"
        },
        {
            "question": "How do I get a new laptop?",
            "expected_response": "REFRESH_AGENT"
        },
        {
            "question": "Laptop refresh",
            "expected_response": "REFRESH_AGENT"
        },
        {
            "question": "My laptop is broken and I need a replacement",
            "expected_response": "REFRESH_AGENT"
        },
        {
            "question": "I need to upgrade my work laptop",
            "expected_response": "REFRESH_AGENT"
        },
        {
            "question": "How can I refresh my company laptop?",
            "expected_response": "REFRESH_AGENT"
        },
        {
            "question": "I want a new laptop for work",
            "expected_response": "REFRESH_AGENT"
        },
        {
            "question": "My laptop needs to be replaced due to hardware issues",
            "expected_response": "REFRESH_AGENT"
        },
        {
            "question": "Can I get a laptop upgrade?",
            "expected_response": "REFRESH_AGENT"
        },
        {
            "question": "Laptop replacement request",
            "expected_response": "REFRESH_AGENT"
        },
        
        # EMAIL_CHANGE_AGENT examples - email changes
        {
            "question": "Can I change my email address",
            "expected_response": "EMAIL_CHANGE_AGENT"
        },
        {
            "question": "I would like to update my email address",
            "expected_response": "EMAIL_CHANGE_AGENT"
        },
        {
            "question": "How do I modify my email in the system?",
            "expected_response": "EMAIL_CHANGE_AGENT"
        },
        {
            "question": "I need to change my work email",
            "expected_response": "EMAIL_CHANGE_AGENT"
        },
        {
            "question": "Can you help me update my email address?",
            "expected_response": "EMAIL_CHANGE_AGENT"
        },
        {
            "question": "Email change request",
            "expected_response": "EMAIL_CHANGE_AGENT"
        },
        {
            "question": "I want to submit an email change",
            "expected_response": "EMAIL_CHANGE_AGENT"
        },
        {
            "question": "My email address needs to be updated",
            "expected_response": "EMAIL_CHANGE_AGENT"
        },
        {
            "question": "How can I change my contact email?",
            "expected_response": "EMAIL_CHANGE_AGENT"
        },
        {
            "question": "I need to modify my email address in my profile",
            "expected_response": "EMAIL_CHANGE_AGENT"
        },
        
        # Other requests that should trigger fallback response
        {
            "question": "Can you help me update ticket 12312",
            "expected_response": "I cannot help you with your request"
        },
        {
            "question": "I need help with password reset",
            "expected_response": "I cannot help you with your request"
        },
        {
            "question": "How do I submit a vacation request?",
            "expected_response": "I cannot help you with your request"
        },
        {
            "question": "Can I get access to the shared drive?",
            "expected_response": "I cannot help you with your request"
        },
        {
            "question": "I need help with my phone setup",
            "expected_response": "I cannot help you with your request"
        },
    ]

    for j in range(10):
        print("")
        print(
            f"Iteration {j} ------------------------------------------------------------"
        )

        # Create a session that will be used to ask the agent a sequence of questions
        session_create_response = client.agents.session.create(
            agent_id, session_name="agent1"
        )
        session_id = session_create_response.session_id

        for i, question_item in enumerate(questions):
            question = question_item["question"]
            expected_response = question_item["expected_response"]
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
            # Check if response matches expected
            response_clean = response.strip()
            expected_clean = expected_response.strip()
            
            if response_clean == expected_clean:
                status = "✓ MATCH"
            elif expected_clean in response_clean:
                status = "~ PARTIAL MATCH"
            else:
                status = "✗ NO MATCH"
            
            print("  STATUS: " + status + " - EXPECTED: " + expected_response + " - RESPONSE:" + response)


if __name__ == "__main__":
    main()
