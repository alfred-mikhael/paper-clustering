from typing import Any, Protocol


class DatabaseClient(Protocol):
    def insert(table: str, records: list[dict[str, Any]]) -> bool: ...

    def select(
        table: str, cols: list[str], where=list[tuple[str, str, Any]]
    ) -> list[list]: ...

    def execute_rpc(
        rpc_name: str, arguments: dict[str, Any]
    ) -> list[dict[str, Any]]: ...
