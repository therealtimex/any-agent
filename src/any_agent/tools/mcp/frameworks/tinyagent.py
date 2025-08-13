"""MCP adapter for Tiny framework."""

import os
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from contextlib import suppress
from datetime import timedelta
from typing import Any, Literal

from pydantic import Field, PrivateAttr

from any_agent.config import AgentFramework, MCPSse, MCPStdio, MCPStreamableHttp, Tool
from any_agent.tools.mcp.mcp_connection import _MCPConnection
from any_agent.tools.mcp.mcp_server import _MCPServerBase

# Check for MCP dependencies
mcp_available = False
with suppress(ImportError):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.sse import sse_client
    from mcp.client.stdio import stdio_client
    from mcp.client.streamable_http import streamablehttp_client
    from mcp.types import Tool as MCPTool  # noqa: TC002

    mcp_available = True


class TinyAgentMCPConnection(_MCPConnection["MCPTool"], ABC):
    """Base class for TinyAgent MCP connections."""

    _client: Any | None = PrivateAttr(default=None)
    _session: ClientSession | None = None

    @abstractmethod
    async def list_tools(self) -> list["MCPTool"]:
        """List tools from the MCP server."""
        if not self._client:
            msg = "MCP client is not set up. Please call `setup` from a concrete class."
            raise ValueError(msg)

        # Get the available tools from the MCP server using schema
        available_tools = await self._session.list_tools()  # type: ignore[union-attr]

        # Filter tools if specific tools were requested
        filtered_tools = self._filter_tools(available_tools.tools)

        # Create callable tool functions
        tool_list = list[Any]()
        for tool_info in filtered_tools:
            tool_list.append(self._create_tool_from_info(tool_info, self._session))  # type: ignore[arg-type]

        return tool_list

    def _create_tool_from_info(
        self, tool: Tool, session: "ClientSession"
    ) -> Callable[..., Any]:
        """Create a tool function from tool information."""
        tool_name = tool.name if hasattr(tool, "name") else tool
        tool_description = tool.description if hasattr(tool, "description") else ""
        input_schema = tool.inputSchema if hasattr(tool, "inputSchema") else None
        if not session:
            msg = "Not connected to MCP server"
            raise ValueError(msg)

        async def tool_function(*args, **kwargs) -> Any:  # type: ignore[no-untyped-def]
            """Tool function that calls the MCP server."""
            # Combine args and kwargs
            combined_args = {}
            if args and len(args) > 0:
                combined_args = args[0]
            combined_args.update(kwargs)

            if not session:
                msg = "Not connected to MCP server"
                raise ValueError(msg)
            # Call the tool on the MCP server
            try:
                return await session.call_tool(tool_name, combined_args)  # type: ignore[arg-type]
            except Exception as e:
                return f"Error calling tool {tool_name}: {e!s}"

        # Set attributes for the tool function
        tool_function.__name__ = tool_name  # type: ignore[assignment]
        tool_function.__doc__ = tool_description
        # this isn't a defined attribute of a callable, but we pass it to tinyagent so that it can use it appropriately
        # when constructing the ToolExecutor.
        tool_function.__input_schema__ = input_schema  # type: ignore[attr-defined]

        return tool_function


class TinyAgentMCPStdioConnection(TinyAgentMCPConnection):
    mcp_tool: MCPStdio

    async def list_tools(self) -> list["MCPTool"]:
        """List tools from the MCP server."""
        server_params = StdioServerParameters(
            command=self.mcp_tool.command,
            args=list(self.mcp_tool.args),
            env={**os.environ,**self.mcp_tool.env},
        )

        self._client = stdio_client(server_params)

        # Setup the client connection using exit stack to manage lifecycle
        read, write = await self._exit_stack.enter_async_context(self._client)

        # Create a client session
        client_session = ClientSession(
            read,
            write,
            timedelta(seconds=self.mcp_tool.client_session_timeout_seconds)
            if self.mcp_tool.client_session_timeout_seconds
            else None,
        )

        # Start the session
        self._session = await self._exit_stack.enter_async_context(client_session)
        if not self._session:
            msg = "Failed to create MCP session"
            raise ValueError(msg)

        await self._session.initialize()

        return await super().list_tools()


class TinyAgentMCPSseConnection(TinyAgentMCPConnection):
    mcp_tool: MCPSse

    async def list_tools(self) -> list["MCPTool"]:
        """List tools from the MCP server."""
        kwargs = {}
        if self.mcp_tool.client_session_timeout_seconds:
            kwargs["sse_read_timeout"] = self.mcp_tool.client_session_timeout_seconds
        self._client = sse_client(
            url=self.mcp_tool.url,
            headers=dict(self.mcp_tool.headers or {}),
            **kwargs,  # type: ignore[arg-type]
        )

        # Setup the client connection using exit stack to manage lifecycle
        read, write = await self._exit_stack.enter_async_context(self._client)

        # Create a client session
        client_session = ClientSession(
            read,
            write,
            timedelta(seconds=self.mcp_tool.client_session_timeout_seconds)
            if self.mcp_tool.client_session_timeout_seconds
            else None,
        )

        # Start the session
        self._session = await self._exit_stack.enter_async_context(client_session)
        if not self._session:
            msg = "Failed to create MCP session"
            raise ValueError(msg)

        await self._session.initialize()

        return await super().list_tools()


class TinyAgentMCPStreamableHttpConnection(TinyAgentMCPConnection):
    mcp_tool: MCPStreamableHttp
    _get_session_id_callback: Callable[[], str | None] | None = None

    def get_session_id(self) -> str | None:
        """Retrieve session ID, if it has been established."""
        if self._get_session_id_callback:
            return self._get_session_id_callback()
        return None

    async def list_tools(self) -> list["MCPTool"]:
        """List tools from the MCP server."""
        kwargs = {}
        if self.mcp_tool.client_session_timeout_seconds:
            kwargs["sse_read_timeout"] = self.mcp_tool.client_session_timeout_seconds
        self._client = streamablehttp_client(
            url=self.mcp_tool.url,
            headers=dict(self.mcp_tool.headers or {}),
            **kwargs,  # type: ignore[arg-type]
        )

        # Setup the client connection using exit stack to manage lifecycle
        streamablehttp_transport = await self._exit_stack.enter_async_context(
            self._client
        )
        read, write, get_session_id_callback = streamablehttp_transport
        self._get_session_id_callback = get_session_id_callback

        # Create a client session
        client_session = ClientSession(
            read,
            write,
            timedelta(seconds=self.mcp_tool.client_session_timeout_seconds)
            if self.mcp_tool.client_session_timeout_seconds
            else None,
        )

        # Start the session
        self._session = await self._exit_stack.enter_async_context(client_session)
        if not self._session:
            msg = "Failed to create MCP session"
            raise ValueError(msg)

        await self._session.initialize()

        return await super().list_tools()


class TinyAgentMCPServerBase(_MCPServerBase["MCPTool"], ABC):
    framework: Literal[AgentFramework.TINYAGENT] = AgentFramework.TINYAGENT
    tools: Sequence["MCPTool"] = Field(default_factory=list)
    libraries: str = "any-agent[mcp]"

    def _check_dependencies(self) -> None:
        """Check if the required dependencies for the MCP server are available."""
        self.mcp_available = mcp_available
        super()._check_dependencies()


class TinyAgentMCPServerStdio(TinyAgentMCPServerBase):
    mcp_tool: MCPStdio

    async def _setup_tools(
        self, mcp_connection: _MCPConnection["MCPTool"] | None = None
    ) -> None:
        mcp_connection = mcp_connection or TinyAgentMCPStdioConnection(
            mcp_tool=self.mcp_tool
        )
        await super()._setup_tools(mcp_connection)


class TinyAgentMCPServerSse(TinyAgentMCPServerBase):
    mcp_tool: MCPSse

    async def _setup_tools(
        self, mcp_connection: _MCPConnection["MCPTool"] | None = None
    ) -> None:
        mcp_connection = mcp_connection or TinyAgentMCPSseConnection(
            mcp_tool=self.mcp_tool
        )
        await super()._setup_tools(mcp_connection)


class TinyAgentMCPServerStreamableHttp(TinyAgentMCPServerBase):
    mcp_tool: MCPStreamableHttp

    async def _setup_tools(
        self, mcp_connection: _MCPConnection["MCPTool"] | None = None
    ) -> None:
        mcp_connection = mcp_connection or TinyAgentMCPStreamableHttpConnection(
            mcp_tool=self.mcp_tool
        )
        await super()._setup_tools(mcp_connection)


TinyAgentMCPServer = (
    TinyAgentMCPServerStdio | TinyAgentMCPServerSse | TinyAgentMCPServerStreamableHttp
)
