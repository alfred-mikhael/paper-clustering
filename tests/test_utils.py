"""Tests for iterable batching and retry behavior without real delays."""

from itertools import count
import unittest
from unittest.mock import Mock, call, patch

from paper_clustering.utils import iter_batches, retry


class IterBatchesTests(unittest.TestCase):
    def test_batch_boundaries_preserve_order(self):
        for items, size, expected in (
            ([], 2, []),
            ([0, 1, 2, 3], 2, [[0, 1], [2, 3]]),
            ([0, 1, 2, 3, 4], 2, [[0, 1], [2, 3], [4]]),
            ([0, 1], 4, [[0, 1]]),
            ([0, 1], 1, [[0], [1]]),
        ):
            with self.subTest(items=items, size=size):
                self.assertEqual(list(iter_batches(iter(items), size)), expected)

    def test_infinite_iterator_is_consumed_one_batch_at_a_time(self):
        items = count()
        batches = iter_batches(items, 3)
        self.assertEqual(next(items), 0)  # Constructing the helper consumes nothing.
        self.assertEqual(next(batches), [1, 2, 3])
        self.assertEqual(next(items), 4)  # No next-batch prefetch.
        self.assertEqual(next(batches), [5, 6, 7])

    def test_invalid_size_does_not_consume_input(self):
        for size in (0, -1, 1.5, True, "2"):
            with self.subTest(size=size):
                items = iter([1, 2])
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    next(iter_batches(items, size))
                self.assertEqual(next(items), 1)

    def test_input_errors_propagate(self):
        def failing_input():
            yield 1
            yield 2
            raise RuntimeError("input failed")

        batches = iter_batches(failing_input(), 2)
        self.assertEqual(next(batches), [1, 2])
        with self.assertRaisesRegex(RuntimeError, "input failed"):
            next(batches)


class RetryTests(unittest.TestCase):
    def setUp(self):
        sleep_patch = patch("paper_clustering.utils.sleep")
        self.sleep = sleep_patch.start()
        self.addCleanup(sleep_patch.stop)
        log_patch = patch("paper_clustering.utils.logger")
        self.logger = log_patch.start()
        self.addCleanup(log_patch.stop)

    def test_success_preserves_result_arguments_and_metadata(self):
        @retry()
        def add(value, *, increment=1):
            """Add an increment."""
            return value + increment

        self.assertEqual(add(2, increment=3), 5)
        self.assertEqual(add.__name__, "add")
        self.assertEqual(add.__doc__, "Add an increment.")
        self.assertEqual(add.__wrapped__(2), 3)
        self.sleep.assert_not_called()

    def test_retries_then_returns_success(self):
        operation = Mock(side_effect=[OSError("first"), OSError("second"), "done"])

        @retry(retries=2, initial_backoff=0.5, jitter=False)
        def execute():
            return operation()

        with patch("paper_clustering.utils.random.random") as random:
            self.assertEqual(execute(), "done")
            random.assert_not_called()
        self.assertEqual(operation.call_count, 3)
        self.assertEqual(self.sleep.call_args_list, [call(0.5), call(1.0)])
        self.assertEqual(
            [entry.args[2] for entry in self.logger.exception.call_args_list], [2, 1]
        )

    def test_exhaustion_raises_last_error_without_extra_sleep(self):
        errors = [OSError("first"), OSError("second"), OSError("last")]
        operation = Mock(side_effect=errors)

        @retry(retries=2, initial_backoff=1, exponential_backoff=False, jitter=False)
        def execute():
            operation()

        with self.assertRaises(OSError) as caught:
            execute()
        self.assertIs(caught.exception, errors[-1])
        self.assertEqual(operation.call_count, 3)
        self.assertEqual(self.sleep.call_args_list, [call(1), call(1)])

    def test_state_resets_between_invocations(self):
        operation = Mock(side_effect=[OSError(), "one", OSError(), "two"])

        @retry(retries=1, initial_backoff=2, jitter=False)
        def execute():
            return operation()

        self.assertEqual(execute(), "one")
        self.assertEqual(execute(), "two")
        self.assertEqual(self.sleep.call_args_list, [call(2), call(2)])

    def test_non_retryable_exceptions_propagate_immediately(self):
        for error in (TypeError("type"), ValueError("value"), KeyboardInterrupt()):
            with self.subTest(error=type(error)):
                operation = Mock(side_effect=error)

                @retry()
                def execute():
                    operation()

                with self.assertRaises(type(error)) as caught:
                    execute()
                self.assertIs(caught.exception, error)
                operation.assert_called_once()
        self.sleep.assert_not_called()

    def test_zero_retries_calls_once(self):
        operation = Mock(side_effect=OSError("failed"))

        @retry(retries=0)
        def execute():
            operation()

        with self.assertRaises(OSError):
            execute()
        operation.assert_called_once()
        self.sleep.assert_not_called()

    def test_jitter_updates_the_next_backoff(self):
        operation = Mock(side_effect=[OSError(), OSError(), OSError(), "done"])

        @retry(retries=3, initial_backoff=1)
        def execute():
            return operation()

        with patch("paper_clustering.utils.random.random", return_value=0.25):
            self.assertEqual(execute(), "done")
        self.assertEqual(self.sleep.call_args_list, [call(1), call(2.25), call(4.75)])

    def test_invalid_retry_settings_are_rejected(self):
        for retries in (-1, 1.5, True):
            with self.subTest(retries=retries):
                with self.assertRaisesRegex(ValueError, "non-negative integer"):
                    retry(retries=retries)
        for backoff in (-1, float("nan"), float("inf")):
            with self.subTest(backoff=backoff):
                with self.assertRaisesRegex(ValueError, "finite and non-negative"):
                    retry(initial_backoff=backoff)


if __name__ == "__main__":
    unittest.main()
