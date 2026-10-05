from typing import Any
from contextlib import AbstractContextManager

import psycopg
from psycopg import sql
from psycopg.rows import dict_row
import logging

logger = logging.getLogger(__name__)


class PostgresClient:
    def __init__(self, conn: psycopg.Connection):
        self.conn = conn
        # If this is not included, need to manually commit and rollback.
        self.conn.autocommit = True

    def transaction(self) -> AbstractContextManager[psycopg.Transaction]:
        """Group operations atomically, even with autocommit enabled."""
        return self.conn.transaction()

    def insert(self, table: str, records: list[dict[str, Any]]) -> bool:
        if not records:
            return True

        columns = list(records[0])
        if not columns:
            raise ValueError("Records must contain at least one column.")

        expected_columns = set(columns)
        for index, record in enumerate(records):
            if set(record) != expected_columns:
                raise ValueError(
                    f"Record {index} does not contain the same columns as record 0."
                )

        query = sql.SQL("INSERT INTO {} ({}) VALUES ({})").format(
            sql.Identifier(table),
            sql.SQL(", ").join(map(sql.Identifier, columns)),
            sql.SQL(", ").join(sql.Placeholder() for _ in columns),
        )
        values = [tuple(record[column] for column in columns) for record in records]

        with self.conn.cursor() as cur:
            cur.executemany(query, values)

        logger.info(f"Inserted {len(records)} records into table {table}")
        return True

    def select(
        self,
        table: str,
        cols: list[str],
        where: list[tuple[str, str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        selected_columns = sql.SQL(", ").join(map(sql.Identifier, cols))
        if cols == ["*"]:
            selected_columns = sql.SQL("*")

        query = sql.SQL("SELECT {} FROM {}").format(
            selected_columns,
            sql.Identifier(table),
        )
        parameters: list[Any] = []
        conditions: list[sql.Composable] = []

        for column, operator, value in where or []:
            if operator != "eq":
                raise ValueError(f"Unsupported WHERE operator: {operator!r}")

            conditions.append(
                sql.SQL("{} = {}").format(
                    sql.Identifier(column),
                    sql.Placeholder(),
                )
            )
            parameters.append(value)

        if conditions:
            query += sql.SQL(" WHERE ") + sql.SQL(" AND ").join(conditions)

        with self.conn.cursor(row_factory=dict_row) as cur:
            cur.execute(query, parameters)
            res = cur.fetchall()
            logger.info(
                f"Retrieved {len(res)} records from table {table} with query {query}"
            )
            return res

    def execute_rpc(self, rpc_name: str, args: dict[str, Any]) -> list[dict[str, Any]]:
        arguments = sql.SQL(", ").join(
            sql.SQL("{} => {}").format(
                sql.Identifier(argument_name),
                sql.Placeholder(argument_name),
            )
            for argument_name in args
        )
        query = sql.SQL("SELECT * FROM {}({})").format(
            sql.Identifier(rpc_name),
            arguments,
        )

        with self.conn.cursor(row_factory=dict_row) as cur:
            cur.execute(query, args)
            return cur.fetchall()
