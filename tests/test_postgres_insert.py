"""PostgreSQL insert contract with a mocked connection, when psycopg is installed."""

import importlib.util
import unittest
from unittest.mock import MagicMock

if importlib.util.find_spec("psycopg") is not None:
    from paper_clustering.postgres import PostgresClient
else:
    PostgresClient = None


@unittest.skipIf(PostgresClient is None, "psycopg is not installed")
class PostgresInsertTests(unittest.TestCase):
    def setUp(self):
        self.connection = MagicMock()
        self.client = PostgresClient(self.connection)

    def test_empty_input_is_no_op_and_valid_input_returns_none(self):
        self.assertIsNone(self.client.insert("papers", []))
        self.connection.cursor.assert_not_called()
        self.assertIsNone(self.client.insert("papers", [{"arxiv_id": "a"}]))
        self.connection.cursor.return_value.__enter__.return_value.executemany.assert_called_once()

    def test_database_errors_propagate(self):
        error = RuntimeError("insertion failed")
        cursor = self.connection.cursor.return_value.__enter__.return_value
        cursor.executemany.side_effect = error
        with self.assertRaises(RuntimeError) as caught:
            self.client.insert("papers", [{"arxiv_id": "a"}])
        self.assertIs(caught.exception, error)

    def test_cluster_upserts_use_existing_unique_keys(self):
        cases = (
            ("cluster_metadata", {"id": 1, "label": "Topic", "size": 2, "parent_cluster_id": -1},
             'ON CONFLICT ("id")', ('"label"', '"size"', '"parent_cluster_id"')),
            ("clusters", {"arxiv_id": "a", "cluster_id": 1, "title": "Title", "certainty": 0.5},
             'ON CONFLICT ("arxiv_id", "cluster_id")', ('"title"', '"certainty"')),
        )
        cursor = self.connection.cursor.return_value.__enter__.return_value
        for table, record, conflict, columns in cases:
            with self.subTest(table=table):
                self.client.insert(table, [record])
                query, values = cursor.executemany.call_args.args
                statement = query.as_string()
                self.assertIn(conflict, statement)
                for column in columns:
                    self.assertIn(f"{column} = EXCLUDED.{column}", statement)
                self.assertEqual(values, [tuple(record.values())])

    def test_inconsistent_records_fail_before_execution(self):
        for records in ([{}], [{"arxiv_id": "a"}, {"title": "B"}]):
            with self.subTest(records=records), self.assertRaises(ValueError):
                self.client.insert("papers", records)
        self.connection.cursor.assert_not_called()

    def test_read_parameters_are_bound_and_identifiers_quoted(self):
        cursor = self.connection.cursor.return_value.__enter__.return_value
        cursor.fetchall.return_value = [{"arxiv_id": "a"}]
        value = "a'; DROP TABLE papers; --"
        self.assertEqual(
            self.client.select('paper"table', ["arxiv_id"], [("arxiv_id", "eq", value)]),
            [{"arxiv_id": "a"}],
        )
        query, parameters = cursor.execute.call_args.args
        self.assertIn('FROM "paper""table"', query.as_string())
        self.assertNotIn(value, query.as_string())
        self.assertEqual(parameters, [value])
        self.client.select("papers", ["*"])
        self.assertIn("SELECT * FROM", cursor.execute.call_args.args[0].as_string())
        with self.assertRaisesRegex(ValueError, "Unsupported WHERE"):
            self.client.select("papers", ["*"], [("id", "raw SQL", "anything")])

    def test_rpc_binds_named_arguments_and_returns_rows(self):
        cursor = self.connection.cursor.return_value.__enter__.return_value
        cursor.fetchall.return_value = [{"arxiv_id": "a"}]
        arguments = {"query_vector": [0.1, 0.2], "match_count": 3}
        self.assertEqual(self.client.execute_rpc("ann", arguments), [{"arxiv_id": "a"}])
        query, parameters = cursor.execute.call_args.args
        self.assertIn('"query_vector" => %(query_vector)s', query.as_string())
        self.assertEqual(parameters, arguments)


if __name__ == "__main__":
    unittest.main()
