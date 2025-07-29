import sys
import json
import logging
import argparse
import asyncio
import random
from datetime import datetime
from typing import Dict
from pydantic import BaseModel
from fastmcp import FastMCP
from llama_stack_client import LlamaStackClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LaptopRequestResponse(BaseModel):
    """Model for laptop request response"""

    employee_id: str
    laptop_model: str
    ticket_number: str
    status: str
    timestamp: str


# Create the FastMCP server
server = FastMCP("ServiceNow Server")


@server.tool()
async def submit_laptop_request(employee_id: str, laptop_model: str) -> str:
    """
    Submit a laptop request to ServiceNow and get a ticket number.

    Args:
        employee_id: The ID of the employee requesting the laptop
        laptop_model: The model of laptop being requested

    Returns:
        JSON string containing ticket information including ticket number, status and timestamp
    """
    # Generate a random ticket number (ServiceNow style)
    ticket_number = f"REQ{random.randint(1000000, 9999999)}"

    # Set status to "Submitted" for new requests
    status = "Submitted"

    laptop_request = LaptopRequestResponse(
        employee_id=employee_id,
        laptop_model=laptop_model,
        ticket_number=ticket_number,
        status=status,
        timestamp=datetime.now().isoformat(),
    )

    logger.info(f"Created laptop request for employee {employee_id}: {laptop_request}")
    return json.dumps(laptop_request.dict())


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="ServiceNow MCP Server")
    parser.add_argument(
        "--llama-stack-host",
        default="localhost:8321",
        help="LLama Stack host and port (default: localhost:8321)",
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="Host IP for this MCP server (default: localhost)",
    )
    parser.add_argument(
        "--port",
        default=8003,
        type=int,
        help="Port for this MCP server (default: 8003)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--no-register",
        action="store_true",
        help="Skip automatic registration with LLama Stack",
    )

    args = parser.parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    logger.info(f"Starting ServiceNow MCP Server")
    logger.info(f"MCP Server host: {args.host}")
    logger.info(f"MCP Server port: {args.port}")
    logger.info(f"LLama Stack host: {args.llama_stack_host}")

    if not args.no_register:
        # Register the MCP toolgroup
        client = LlamaStackClient(
            base_url=f"http://{args.llama_stack_host}:8321",
            timeout=120.0,
        )
        client.toolgroups.register(
            toolgroup_id="mcp::servicenow",
            provider_id="model-context-protocol",
            mcp_endpoint={"uri": f"http://{args.host}:{args.port}/sse"},
        )
        logger.info(f"Registered MCP server at http://{args.host}:{args.port}/sse")

    # Run the FastMCP server with SSE transport
    logger.info("Starting FastMCP server with SSE transport...")
    server.run(transport="sse", host=args.host, port=args.port)


if __name__ == "__main__":
    exit_code = main()
    if exit_code:
        sys.exit(exit_code)
