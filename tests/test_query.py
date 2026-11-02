"""Search results are ranked per paper while retaining matching passage pairs."""

import unittest
from unittest.mock import Mock

from paper_clustering.query import SearchResult, find_similar, rank_candidates, retrieve_candidates


class QueryTests(unittest.TestCase):
    def setUp(self):
        self.candidates = [
            SearchResult("query", "a", "A", "q1", "a1", 0.5),
            SearchResult("query", "b", "B", "q1", "b1", 0.9),
            SearchResult("query", "a", "A", "q2", "a2", 0.8),
            SearchResult("query", "c", "C", "q2", "c1", 0.7),
        ]

    def test_similarity_ranking_limits_papers_and_retains_all_their_pairs(self):
        result = rank_candidates(self.candidates, k=2)
        self.assertEqual(list(result), ["b", "a"])
        self.assertEqual(result["a"], [self.candidates[0], self.candidates[2]])
        self.assertEqual(rank_candidates(self.candidates, k=0), {})

    def test_reranker_orders_papers_by_best_pair_without_replacing_similarity(self):
        reranker = Mock()
        reranker.score_pairs.return_value = [0.95, 0.1, 0.3, 0.8]
        result = rank_candidates(self.candidates, k=2, reranker=reranker)
        self.assertEqual(list(result), ["a", "c"])
        self.assertEqual([pair.similarity for pair in result["a"]], [0.5, 0.8])
        reranker.score_pairs.assert_called_once_with(
            [(pair.query_passage, pair.target_passage) for pair in self.candidates]
        )

    def test_reranker_ties_keep_input_order_and_bad_score_counts_fail(self):
        reranker = Mock()
        reranker.score_pairs.return_value = [1] * 4
        self.assertEqual(list(rank_candidates(self.candidates, reranker=reranker)), ["a", "b", "c"])
        reranker.score_pairs.return_value = [1]
        with self.assertRaisesRegex(ValueError, "one score"):
            rank_candidates(self.candidates, reranker=reranker)

    def test_empty_candidates_do_not_call_reranker(self):
        reranker = Mock()
        self.assertEqual(rank_candidates([], reranker=reranker), {})
        reranker.score_pairs.assert_not_called()

    def test_retrieval_excludes_query_paper_and_preserves_passage_context(self):
        client = Mock()
        client.select.return_value = [
            {"arxiv_id": "query", "embedding": [1, 0], "passage": "q1"},
            {"arxiv_id": "query", "embedding": [0, 1], "passage": "q2"},
        ]
        client.execute_rpc.return_value = [
            {"arxiv_id": "query", "title": "Query", "passage": "self", "similarity": 1},
            {"arxiv_id": "a", "title": "A", "passage": "match", "similarity": 0.7},
        ]
        results = find_similar("query", client, k=1)
        self.assertEqual(list(results), ["a"])
        self.assertEqual([pair.query_passage for pair in results["a"]], ["q1", "q2"])
        self.assertEqual(client.execute_rpc.call_args_list[0].args,
                         ("ann", {"query_vector": [1, 0], "match_count": 2}))

    def test_paper_without_passages_never_queries_vectors(self):
        client = Mock()
        client.select.return_value = []
        self.assertEqual(retrieve_candidates("missing", client), [])
        client.execute_rpc.assert_not_called()


if __name__ == "__main__":
    unittest.main()
