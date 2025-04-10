import asyncio
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
from openai import OpenAI
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

client = OpenAI(api_key="FAKEAPI", base_url="http://103.78.3.96:8000/v1")
model= "meta-llama/Llama-3.1-8B-Instruct"


class ConnectionManager:
    def __init__(self, sse_server_map):
        self.sse_server_map = sse_server_map
        self.sessions = {}
        self.exit_stack = AsyncExitStack()

    async def initialize(self):
        # Initialize SSE connections
        for server_name, url in self.sse_server_map.items():
            sse_transport = await self.exit_stack.enter_async_context(
                sse_client(url=url)
            )
            read, write = sse_transport
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await session.initialize()
            self.sessions[server_name] = session

    async def list_tools(self):
        tool_map = {}
        consolidated_tools = []
        for server_name, session in self.sessions.items():
            tools = await session.list_tools()
            tool_map.update({tool.name: server_name for tool in tools.tools})
            consolidated_tools.extend(tools.tools)
        return tool_map, consolidated_tools
    
    async def list_resources(self):
        resource_map = {}
        consolidated_resources = []
        for server_name, session in self.sessions.items():
            resources = await session.list_resources()
            # Assuming resources.resources returns a list of resource objects with uri attribute
            resource_map.update({
                resource.uri.split('://')[1].split('/')[0]: server_name 
                for resource in resources.resources
            })
            consolidated_resources.extend(resources.resources)
        return resource_map, consolidated_resources

    async def call_tool(self, tool_name, arguments, tool_map):
        server_name = tool_map.get(tool_name)
        if not server_name:
            print(f"Tool '{tool_name}' not found.")
            return

        session = self.sessions.get(server_name)
        if session:
            result = await session.call_tool(tool_name, arguments=arguments)
            return result.content[0].text
    
    async def call_resource(self, resource_uri, parameters, resource_map):
        # Parse resource URI to extract the base name
        parts = resource_uri.split('://')
        if len(parts) != 2:
            print(f"Invalid resource URI format: {resource_uri}")
            return
        
        scheme, path = parts
        resource_base = path.split('/')[0]  # Get the base resource name
        
        server_name = resource_map.get(resource_base)
        if not server_name:
            print(f"Resource '{resource_base}' not found.")
            return

        session = self.sessions.get(server_name)
        if session:
            # Call the resource with the URI and parameters
            result = await session.call_resource(resource_uri, parameters=parameters)
            return result.content[0].text

    async def close(self):
        await self.exit_stack.aclose()


# Chat function to handle interactions with tools and resources
async def chat(
    input_messages,
    tool_map,
    resource_map,
    tools,
    resources,
    max_turns=3,
    connection_manager=None,
):
    chat_messages = input_messages[:]
    for _ in range(max_turns):
        result = client.chat.completions.create(
            model=model,
            messages=chat_messages,
            tools=tools,
            resources=resources,  # Add resources to the API call
        )

        # Handle tool calls
        if result.choices[0].finish_reason == "tool_calls":
            chat_messages.append(result.choices[0].message)
            # Loop and call and append to message array
            for tool_call in result.choices[0].message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                # Get server name for the tool just for logging
                server_name = tool_map.get(tool_name, "")

                # Log tool call
                log_message = f"**Tool Call**  \n**Tool Name:** `{tool_name}` from **MCP Server**: `{server_name}`  \n**Input:**  \n```json\n{json.dumps(tool_args, indent=2)}\n```"
                yield {"role": "assistant", "content": log_message}

                # Call the tool and log its observation
                observation = await connection_manager.call_tool(
                    tool_name, tool_args, tool_map
                )
                log_message = f"**Tool Observation**  \n**Tool Name:** `{tool_name}` from **MCP Server**: `{server_name}`  \n**Output:**  \n```json\n{json.dumps(observation, indent=2)}\n```  \n---"
                yield {"role": "assistant", "content": log_message}

                chat_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(observation),
                    }
                )
        
        # Handle resource calls
        elif result.choices[0].message.get("resource_calls"):
            chat_messages.append(result.choices[0].message)
            for resource_call in result.choices[0].message.resource_calls:
                resource_uri = resource_call.uri
                resource_params = resource_call.parameters
                
                # Extract resource base name for mapping
                resource_base = resource_uri.split('://')[1].split('/')[0]
                server_name = resource_map.get(resource_base, "")
                
                # Log resource call
                log_message = f"**Resource Call**  \n**Resource URI:** `{resource_uri}` from **MCP Server**: `{server_name}`  \n**Parameters:**  \n```json\n{json.dumps(resource_params, indent=2)}\n```"
                yield {"role": "assistant", "content": log_message}
                
                # Call the resource
                response = await connection_manager.call_resource(
                    resource_uri, resource_params, resource_map
                )
                
                # Log response
                log_message = f"**Resource Response**  \n**Resource URI:** `{resource_uri}`  \n**Output:**  \n```json\n{json.dumps(response, indent=2)}\n```  \n---"
                yield {"role": "assistant", "content": log_message}
                
                # Add to chat history
                chat_messages.append({
                    "role": "resource",
                    "resource_call_id": resource_call.id,
                    "content": str(response)
                })
        
        # Handle regular message response
        else:
            yield {"role": "assistant", "content": result.choices[0].message.content}
            return

    # Generate a final response if max turns are reached
    print("✅Before ", chat_messages)
    result = client.chat.completions.create(
        model=model,
        messages=chat_messages,
    )
    yield {"role": "assistant", "content": result.choices[0].message.content}


if __name__ == "__main__":
    sse_server_map = {
        "python_executor_mcp": "http://localhost:8000/sse",
    }

    async def main():
        connection_manager = ConnectionManager(sse_server_map)
        await connection_manager.initialize()

        # Get tools and create tool map
        tool_map, tool_objects = await connection_manager.list_tools()
        tools_json = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "strict": True,
                    "parameters": tool.inputSchema,
                },
            }
            for tool in tool_objects
        ]
        
        # Get resources and create resource map
        resource_map, resource_objects = await connection_manager.list_resources()
        resources_json = [
            {
                "type": "resource",
                "resource": {
                    "uri": resource.uri,
                    "description": resource.description,
                    "strict": True,
                    "parameters": resource.inputSchema,
                },
            }
            for resource in resource_objects
        ]

        input_messages = [
            {
                "role": "system",
                "content": "Use the tools and resources to answer the questions.",
            },
            {"role": "user", "content": "What is temperature in Hanoi?"},
        ]

        async for response in chat(
            input_messages,
            tool_map,
            resource_map,
            tools=tools_json,
            resources=resources_json,
            connection_manager=connection_manager,
        ):
            print(response)

        await connection_manager.close()

    asyncio.run(main())
