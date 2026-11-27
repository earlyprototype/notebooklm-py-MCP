"""Exercise the installed console script from a temporary, credential-free home."""

import asyncio
import shutil
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    command = shutil.which("notebooklm-mcp", path=str(Path(sys.executable).parent))
    assert command, "Missing notebooklm-mcp console script in the test environment"
    expected = (Path(__file__).resolve().parents[2] / "INSTRUCTIONS.md").read_text(encoding="utf-8")
    with TemporaryDirectory() as directory:
        env = {"HOME": directory, "USERPROFILE": directory, "NOTEBOOKLM_HOME": str(Path(directory) / ".notebooklm")}
        params = StdioServerParameters(command=command, cwd=directory, env=env)
        async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
            initialized = await session.initialize()
            tools = await session.list_tools()
            assert len({tool.name for tool in tools.tools}) == 72
            assert initialized.instructions == expected
    print("Installed stdio server listed 72 tools and returned the full INSTRUCTIONS.md.")


if __name__ == "__main__":
    asyncio.run(asyncio.wait_for(main(), timeout=60))
