from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import PurePath


class ActionKind(str, Enum):
    LLM_TEXT = "llm_text"
    LLM_TOOL_CALL = "llm_tool_call"
    TOOL_SHELL = "tool_shell"
    TOOL_SEARCH = "tool_search"
    TOOL_FILE = "tool_file"
    TOOL_BROWSER = "tool_browser"
    TOOL_OTHER = "tool_other"
    SPAWN_CHILD = "spawn_child"
    WAIT_CHILD = "wait_child"
    WAIT_JOIN = "wait_join"
    MESSAGE = "message"
    RETURN = "return"


@dataclass(frozen=True)
class NormalizedTool:
    family: str
    backend_class: str


class ToolTaxonomy:
    """Map unstable runtime names to portable, low-cardinality features."""

    _SEARCH_TERMS = {"search", "query", "serp", "retrieval", "retrieve"}
    _BROWSER_TERMS = {"browser", "browse", "fetch", "http", "web", "url"}
    _FILE_TERMS = {"file", "read", "write", "edit", "glob", "grep"}
    _SHELL_TERMS = {"shell", "bash", "terminal", "command", "exec", "run"}

    def __init__(self, aliases: dict[str, str] | None = None) -> None:
        self.aliases = {key.lower(): value.lower() for key, value in (aliases or {}).items()}

    def normalize(self, tool_name: str, endpoint: str | None = None) -> NormalizedTool:
        name = tool_name.strip().lower()
        family = self.aliases.get(name) or self._infer_family(name)
        backend = self._backend_class(endpoint or name)
        return NormalizedTool(family=family, backend_class=backend)

    def action_for_family(self, family: str) -> ActionKind:
        return {
            "shell": ActionKind.TOOL_SHELL,
            "search": ActionKind.TOOL_SEARCH,
            "file": ActionKind.TOOL_FILE,
            "browser": ActionKind.TOOL_BROWSER,
        }.get(family, ActionKind.TOOL_OTHER)

    def _infer_family(self, name: str) -> str:
        terms = set(name.replace("-", "_").split("_"))
        if terms & self._SEARCH_TERMS:
            return "search"
        if terms & self._BROWSER_TERMS:
            return "browser"
        if terms & self._FILE_TERMS:
            return "file"
        if terms & self._SHELL_TERMS:
            return "shell"
        return "other"

    @staticmethod
    def _backend_class(endpoint: str) -> str:
        endpoint = endpoint.strip().lower()
        if not endpoint:
            return "unknown"
        if "://" in endpoint:
            host = endpoint.split("://", 1)[1].split("/", 1)[0]
            labels = host.split(".")
            return ".".join(labels[-2:]) if len(labels) >= 2 else host
        executable = PurePath(endpoint.split()[0]).name
        return executable or "unknown"
