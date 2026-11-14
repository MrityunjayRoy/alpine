from typing import override

from pydantic import BaseModel, Field

from alpine.tools.base_tool import Tool, ToolType, ToolInvocation, ToolResult
from alpine.utils.paths import resolve_path, is_binary_file
from alpine.utils.text import count_token, truncate_text

class ReadFileParams(BaseModel):
    path: str = Field(
            ...,
            description='Path to the file to read(related to working directory or absolute)'
            )
    
    offset: int = Field(
            1,
            ge=1,
            description="line number to start reading from (1-based).Defaults to 1"
            )

    limit: int | None = Field(
            None,
            ge=1,
            description="maximum number of files to read."
            )

class ReadFileTool(Tool):
    name = "read_tool"
    description = (
            "Read the contents of a text file. Returns the file content with line numbers. "
            "For large files, use offset and limit to read specific portions. "
            "Cannot read binary files (images, executables, etc.)."
            )
    kind = ToolType.READ
    schema = ReadFileParams

    MAX_FILE_SIZE = 1024*1024*10
    MAX_OUTPUT_TOKEN = 25000

    @override
    async def execute(self, invocation: ToolInvocation) -> ToolResult:
        params = ReadFileParams(**invocation.params)
        path = resolve_path(invocation.cwd, params.path)

        if not path.exists():
            return ToolResult.err_result(f"File does not exist: {path}")

        if not path.is_file():
            return ToolResult.err_result(f"Path is not a file: {path}")

        file_size = path.stat().st_size

        if file_size > self.MAX_FILE_SIZE:
            return ToolResult.err_result(f"file too large! ({file_size / (1024*1024):.1f}MB).")

        if is_binary_file(path):
           file_size_mb = file_size / (1024*1024) 
           size_str = f"{file_size_mb:.2f}MB" if file_size_mb >= 1 else f"{file_size} bytes"
           return ToolResult.err_result(f"Cannot read binary file: {path.name} ({size_str})")

        try:
            try:
                content = path.read_text(encoding='utf-8')
            except UnicodeDecodeError:
                content = path.read_text(encoding='latin-1')

            lines = content.splitlines()
            total_lines = len(lines)

            if total_lines == 0:
                return ToolResult.success_result(f"File empty", metadata={"lines": 0})

            start_idx = max(0, params.offset - 1)
            if params.limit is not None:
                end_idx = min(start_idx + params.limit, total_lines)
            else:
                end_idx = total_lines

            selected_lines = lines[start_idx:end_idx]
            formatted_lines = []

            for i, line in enumerate(selected_lines, start=start_idx + 1):
                formatted_lines.append(f"{i:6} | {line}")

            output = '\n'.join(formatted_lines)
            token_count = count_token(output)

            truncated = False
            if token_count > self.MAX_OUTPUT_TOKEN:
                output = truncate_text(
                        output,
                        max_token=self.MAX_OUTPUT_TOKEN,
                        suffix=f"\n... [truncated {total_lines} total lines]",
                        )
                truncated = True

            metadata_lines = []
            if start_idx > 0 or end_idx < total_lines:
                metadata_lines.append(
                    f"Showing lines {start_idx+1}-{end_idx} of {total_lines}"
                )

            if metadata_lines:
                header = " | ".join(metadata_lines) + "\n\n"
                output = header + output

            return ToolResult.success_result(
                output=output,
                truncated=truncated,
                metadata={
                    "path": str(path),
                    "total_lines": total_lines,
                    "shown_start": start_idx + 1,
                    "shown_end": end_idx,
                },
            )

        except Exception as e:
            return ToolResult.err_result(f"Failed to read file {e}")
