import sys
import json
import logging
import argparse
import asyncio
from datetime import datetime
from typing import Dict
from pydantic import BaseModel
from fastmcp import FastMCP
from llama_stack_client import LlamaStackClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global counter for alternating purchase dates
purchase_date_counter = 0


class LaptopInfo(BaseModel):
    """Model for laptop information response"""

    employee_id: str
    geo: str
    purchase_date: str
    timestamp: str


# Create the FastMCP server
server = FastMCP("Asset Database Server")


@server.tool()
async def get_laptop_info(employee_id: str) -> str:
    """
    Get laptop information for an employee including geo location and purchase date.

    Args:
        employee_id: The ID of the employee to look up laptop information for

    Returns:
        JSON string containing laptop information including geo, purchase date and timestamp
    """
    global purchase_date_counter

    # Alternate between two purchase dates as specified
    if purchase_date_counter % 2 == 0:
        purchase_date = "5 years 1 month"
    else:
        purchase_date = "2 years 3 months"
    purchase_date_counter += 1

    # Always return "North America" as geo as specified
    geo = "North America"

    laptop_info = LaptopInfo(
        employee_id=employee_id,
        geo=geo,
        purchase_date=purchase_date,
        timestamp=datetime.now().isoformat(),
    )

    logger.info(f"Retrieved laptop info for employee {employee_id}: {laptop_info}")
    return json.dumps(laptop_info.dict())


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Asset Database MCP Server")
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
        default=8002,
        type=int,
        help="Port for this MCP server (default: 8002)",
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

    logger.info(f"Starting Asset DB MCP Server")
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
            toolgroup_id="mcp::asset_db_server",
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
