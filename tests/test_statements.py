"""Statement extraction and proof dependencies from representative LaTeX."""

import unittest

from paper_clustering.data_models import ArxivSection
from paper_clustering.extract.extract_statements import build_statement_dependency, extract_statements


def section(source):
    return ArxivSection("paper", 0, "Details", "Main results", (), source)


class StatementTests(unittest.TestCase):
    def test_custom_starred_and_nested_statements_preserve_bodies_and_direct_labels(self):
        source = r"""\newtheorem{special}{Special}
        \begin{special}\label{outer}Outer
          \begin{lemma*}\label[lemma]{inner}Inner\end{lemma*}
        \end{special}"""
        statements = extract_statements(section(source))
        self.assertEqual([item.statement_type for item in statements], ["special", "lemma*"])
        self.assertEqual([item.labels for item in statements], [("outer",), ("inner",)])
        self.assertEqual(statements[1].raw_latex, r"\label[lemma]{inner}Inner")
        self.assertEqual(statements[0].major_section_header, "Main results")
        self.assertIn(r"\begin{lemma*}", statements[0].raw_latex)

    def test_comments_verbatim_and_unclosed_environments_are_ignored(self):
        source = r"""% \begin{theorem}Comment\end{theorem}
        \begin{verbatim}\begin{lemma}Code\end{lemma}\end{verbatim}
        \begin{theorem}\label{real}Actual statement\end{theorem}
        \begin{lemma}Unclosed"""
        self.assertEqual([item.labels for item in extract_statements(section(source))], [("real",)])

    def test_dependencies_resolve_aliases_deduplicate_and_ignore_self_or_unknown(self):
        source = r"""\begin{lemma}\label{a}\label{alias-a}A\end{lemma}
        \begin{theorem}\label{b}B\end{theorem}
        \begin{proof}Use \cref{alias-a,a,b,missing} and \eqref{a}.\end{proof}"""
        part = section(source)
        self.assertEqual(build_statement_dependency(extract_statements(part), [part]),
                         {"a": [], "b": ["a"]})

    def test_explicit_proof_owner_takes_priority_over_unnamed_proof(self):
        source = r"""\begin{lemma}\label{a}A\end{lemma}
        \begin{theorem}\label{b}B\end{theorem}
        \begin{proof}An unnamed proof citing \ref{a}.\end{proof}
        \begin{lemma}\label{c}C\end{lemma}
        \begin{proof}[Proof of \ref{b}]Uses \autoref{c}.\end{proof}
        \begin{proof}[Proof of \ref{missing}]Uses \ref{a}.\end{proof}"""
        part = section(source)
        self.assertEqual(build_statement_dependency(extract_statements(part), [part]),
                         {"a": [], "b": ["c"], "c": []})

    def test_proof_in_later_section_links_to_preceding_statement(self):
        parts = [section(r"\begin{lemma}\label{a}A\end{lemma}\begin{theorem}\label{b}B\end{theorem}"),
                 section(r"\begin{proof*}Use \Cref*{a}.\end{proof*}")]
        statements = [item for part in parts for item in extract_statements(part)]
        self.assertEqual(build_statement_dependency(statements, parts), {"a": [], "b": ["a"]})


if __name__ == "__main__":
    unittest.main()
