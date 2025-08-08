"""
Complete Laptop Refresh Agent - LangGraph with Auto Tool Discovery and Iteration

This implementation provides:
1. LangGraph StateGraph for conversation flow management
2. Automatic tool discovery and wrapping from LlamaStack 
3. Question iteration pattern like the original sa/run-flow.py

Features:
- Dynamic tool loading and wrapping as LangChain BaseTool
- Proper type handling for all tool parameters
- Structured conversation flow with LangGraph
- Multi-iteration testing with randomized questions
"""

import json
import logging
import random
import time
from typing import Dict, Any, List, TypedDict, Annotated
from datetime import datetime
import os

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from llama_stack_client import LlamaStackClient

# Fix for Pydantic compatibility issues
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

try:
    from langchain_core.language_models import BaseLanguageModel
    ChatOpenAI.model_rebuild()
except Exception:
    pass

logger = logging.getLogger(__name__)

# Suppress external library logging
logging.getLogger("langgraph").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("openai").setLevel(logging.ERROR)


class State(TypedDict):
    """State for the laptop refresh conversation"""
    messages: Annotated[List[BaseMessage], add_messages]


def create_llamastack_tool(tool_def, llama_client: LlamaStackClient) -> BaseTool:
    """Factory function to create a LlamaStack tool wrapper"""
    
    # Create the tool name and description
    tool_name = f"llamastack_{tool_def.identifier}"
    tool_description = tool_def.description or f"Tool {tool_def.identifier}"
    
    # Build detailed description with parameters
    param_descriptions = []
    for param in tool_def.parameters:
        param_desc = f"{param.name} ({param.parameter_type})"
        if param.required:
            param_desc += " [required]"
        if param.description:
            param_desc += f": {param.description}"
        param_descriptions.append(param_desc)
    
    if param_descriptions:
        enhanced_description = tool_description + f"\n\nParameters:\n" + "\n".join(f"- {desc}" for desc in param_descriptions)
    else:
        enhanced_description = tool_description
    
    # Create a custom tool class dynamically
    class DynamicLlamaStackTool(BaseTool):
        name: str = tool_name
        description: str = enhanced_description
        
        def __init__(self):
            super().__init__()
            # Use setattr to bypass Pydantic validation
            object.__setattr__(self, 'tool_def', tool_def)
            object.__setattr__(self, 'llama_client', llama_client)
        
        def _run(self, **kwargs) -> str:
            """Execute the LlamaStack tool with dynamic arguments"""
            try:
                tool_name = self.tool_def.identifier
                logger.debug(f"Calling LlamaStack tool: {tool_name} with args: {kwargs}")
                
                # Handle nested kwargs from LangGraph
                if 'kwargs' in kwargs and len(kwargs) == 1:
                    # LangGraph is passing nested kwargs
                    actual_kwargs = kwargs['kwargs']
                else:
                    actual_kwargs = kwargs
                
                # Convert arguments to proper types based on tool definition
                processed_kwargs = self._process_arguments(actual_kwargs)
                
                # Special handling for knowledge_search tool
                if tool_name == "knowledge_search":
                    processed_kwargs["vector_db_ids"] = ["laptop-refresh-knowledge-base"]
                
                # Call the actual LlamaStack tool
                response = self.llama_client.tool_runtime.invoke_tool(
                    tool_name=tool_name,
                    kwargs=processed_kwargs
                )
                
                logger.debug(f"Tool {tool_name} response: {response}")
                
                # Extract content from the tool response
                if hasattr(response, 'content') and response.content:
                    if isinstance(response.content, list) and len(response.content) > 0:
                        # Handle multiple content items
                        if len(response.content) == 1:
                            return response.content[0].text
                        else:
                            # Concatenate multiple content items
                            return "\n".join(item.text for item in response.content if hasattr(item, 'text'))
                    else:
                        return str(response.content)
                else:
                    # Fallback to string representation
                    if hasattr(response, 'dict'):
                        return json.dumps(response.dict())
                    else:
                        return str(response)
                        
            except Exception as e:
                logger.error(f"Error calling LlamaStack tool {self.tool_def.identifier}: {e}")
                raise e
        
        def _process_arguments(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
            """Process and validate arguments based on tool definition"""
            processed = {}
            
            for param in self.tool_def.parameters:
                param_name = param.name
                if param_name in kwargs:
                    value = kwargs[param_name]
                    
                    # Type conversion based on parameter definition
                    if param.parameter_type == "string":
                        processed[param_name] = str(value)
                    elif param.parameter_type == "integer":
                        processed[param_name] = int(value) if not isinstance(value, int) else value
                    elif param.parameter_type == "number":
                        processed[param_name] = float(value) if not isinstance(value, (int, float)) else value
                    elif param.parameter_type == "boolean":
                        processed[param_name] = bool(value) if not isinstance(value, bool) else value
                    else:
                        processed[param_name] = value
                elif param.required:
                    logger.warning(f"Required parameter {param_name} not provided for tool {self.tool_def.identifier}")
            
            # Add any extra parameters that weren't in the tool definition
            for key, value in kwargs.items():
                if key not in processed:
                    processed[key] = value
                    
            return processed
        
        async def _arun(self, **kwargs) -> str:
            """Async version of the tool"""
            return self._run(**kwargs)
    
    return DynamicLlamaStackTool()


class CompleteLaptopRefreshAgent:
    """Complete LangGraph-based agent with auto tool discovery and iteration"""
    
    def __init__(self, openai_api_key: str = None, model: str = "llama-4-scout-17b-16e-w4a16", llama_stack_host: str = "10.1.2.128:8321"):
        # Initialize the LLM pointing to LlamaStack instance
        self.llm = ChatOpenAI(
            model=model,
            api_key=openai_api_key or "dummy-key-for-llamastack",
            base_url=f"http://{llama_stack_host}/v1/openai/v1",
            temperature=0,
            # Ensure tool calling is enabled
            model_kwargs={"tool_choice": "auto"}
        )
        
        # Initialize LlamaStack client for tool discovery and execution
        self.llama_client = LlamaStackClient(
            base_url=f"http://{llama_stack_host}",
            timeout=120.0,
        )
        
        # Auto-discover and wrap tools from LlamaStack
        self.tools = self._discover_and_wrap_tools()
        logger.debug(f"Auto-discovered and wrapped {len(self.tools)} tools from LlamaStack")
        
        # Read system prompt from file
        with open("prompt.txt", "r") as f:
            self.system_prompt = f.read().strip()
        
        # Create the ReAct agent using LangGraph prebuilt with auto-discovered tools
        # Ensure tools are available or throw error
        if not self.tools:
            raise RuntimeError(
                f"No tools were discovered from LlamaStack at {llama_stack_host}. "
                f"Expected tools: get_laptop_info, submit_laptop_request, knowledge_search. "
                f"Please ensure LlamaStack is properly configured with MCP servers and knowledge base."
            )
        
        self.agent = create_react_agent(
            model=self.llm,
            tools=self.tools,
            state_modifier=self.system_prompt
        )
        
        # Configure agent to run until completion (multiple steps if needed)
        self.agent = self.agent.with_config({"recursion_limit": 10})
        
        # Define the question sequence like in the original sa/run-flow.py
        self.base_questions = [
            "Can I replace my laptop, my employee id is 1234",
            "Yes", 
            "3",  # This will be randomized like in original
            "proceed"
        ]
    
    def _discover_and_wrap_tools(self) -> List[BaseTool]:
        """Automatically discover and wrap tools from LlamaStack"""
        try:
            tools_list = self.llama_client.tools.list()
            target_tools = ['get_laptop_info', 'submit_laptop_request', 'knowledge_search']
            
            wrapped_tools = []
            for tool in tools_list:
                if hasattr(tool, 'identifier') and tool.identifier in target_tools:
                    wrapped_tool = create_llamastack_tool(tool, self.llama_client)
                    wrapped_tools.append(wrapped_tool)
                    logger.debug(f"Auto-wrapped tool: {tool.identifier} -> {wrapped_tool.name}")
            
            return wrapped_tools
            
        except Exception as e:
            logger.error(f"Failed to discover tools from LlamaStack: {e}")
            return []
    
    def run_conversation(self, user_message: str, conversation_history: List[BaseMessage] = None) -> str:
        """Run a conversation turn using the ReAct agent"""
        
        # Build conversation history
        messages = conversation_history or []
        messages.append(HumanMessage(content=user_message))
        
        # Create initial state
        initial_state = {
            "messages": messages
        }
        
        try:
            result = self.agent.invoke(initial_state)
            
            # Return the last AI message
            for message in reversed(result["messages"]):
                if isinstance(message, AIMessage):
                    return message.content
            
            return "I'm here to help with your laptop refresh needs. How can I assist you today?"
            
        except Exception as e:
            logger.error(f"Error in conversation: {e}")
            return f"I apologize, but I encountered an error: {str(e)}. Please try again."
    
    def run_iterations(self, num_iterations: int = 1) -> None:
        """Run multiple iterations like in the original sa/run-flow.py"""
        
        for j in range(num_iterations):
            print("")
            print(f"Iteration {j} ------------------------------------------------------------")
            
            # Create a fresh conversation for each iteration (like original sessions)
            conversation = ConversationManager(self)
            
            # Create questions with randomized laptop choice (like original)
            questions = self.base_questions.copy()
            questions[2] = str(random.randint(1, 5))  # Randomize laptop choice
            
            try:
                for i, question in enumerate(questions):
                    print("QUESTION: " + question)
                    response = conversation.send_message(question)
                    print("  RESPONSE:" + response)
                    
                    # Add delay between questions for readability
                    time.sleep(0.5)
                    
            except Exception as e:
                print(f"Error in iteration {j}: {e}")
                logger.error(f"Iteration {j} failed: {e}")
                
            # Add delay between iterations
            time.sleep(1)


class ConversationManager:
    """Manages conversation state across multiple turns"""
    
    def __init__(self, agent: CompleteLaptopRefreshAgent):
        self.agent = agent
        self.conversation_history: List[BaseMessage] = []
    
    def send_message(self, user_message: str) -> str:
        """Send a message and maintain conversation history"""
        
        # Get response from agent
        response = self.agent.run_conversation(user_message, self.conversation_history.copy())
        
        # Update conversation history
        self.conversation_history.append(HumanMessage(content=user_message))
        self.conversation_history.append(AIMessage(content=response))
        
        logger.debug(f"Conversation history now has {len(self.conversation_history)} messages")
        
        return response
    
    def reset_conversation(self):
        """Reset the conversation history"""
        self.conversation_history = []


def main():
    """Main function with iteration control like original sa/run-flow.py"""
    
    # Configuration from environment or defaults
    llama_stack_host = os.getenv("LLAMA_STACK_HOST", "10.1.2.128:8321")
    model_id = os.getenv("MODEL_ID", "llama-4-scout-17b-16e-w4a16") 
    openai_api_key = os.getenv("OPENAI_API_KEY", "dummy-key-for-llamastack")
    
    # Suppressing verbose startup output to match sa/run-flow.py
    
    try:
        # Initialize the complete agent
        agent = CompleteLaptopRefreshAgent(
            openai_api_key=openai_api_key,
            model=model_id,
            llama_stack_host=llama_stack_host
        )
        
        # Agent initialization complete - suppress verbose output to match sa/run-flow.py
        
        # Run iterations like the original
        agent.run_iterations(num_iterations=3)  # Reduced for testing
        
    except Exception as e:
        print(f"Failed to initialize agent: {e}")
        logger.error(f"Agent initialization failed: {e}")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Suppress noise from HTTP, OpenAI, and LangGraph
    logging.getLogger("httpx").setLevel(logging.ERROR)
    logging.getLogger("openai").setLevel(logging.ERROR)
    logging.getLogger("langgraph").setLevel(logging.ERROR)
    
    main()
