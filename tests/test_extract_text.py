"""Source archive assembly and LaTeX cleaning without downloads."""

import gzip
import io
import tarfile
import unittest
from zipfile import ZipFile

from paper_clustering.extract.extract_text import (
    TexSource, build_tex_source, build_tex_source_from_bytes,
    clean_latex_for_embedding, extract_sections_from_latex, iter_tex_sources,
)


class TextExtractionTests(unittest.TestCase):
    def test_plain_gzip_zip_and_tar_sources(self):
        source = r"\section{Intro}Hello".encode()
        zipped = io.BytesIO()
        with ZipFile(zipped, "w") as archive:
            archive.writestr("main.tex", source)
            archive.writestr("figure.png", b"not TeX")
        tarred = io.BytesIO()
        with tarfile.open(fileobj=tarred, mode="w:gz") as archive:
            for name, content in (("main.tex", source), ("figure.png", b"image")):
                member = tarfile.TarInfo(name)
                member.size = len(content)
                archive.addfile(member, io.BytesIO(content))
        for payload in (source, gzip.compress(source), zipped.getvalue(), tarred.getvalue()):
            with self.subTest(format=payload[:8]):
                sources = list(iter_tex_sources(payload))
                self.assertEqual(len(sources), 1)
                self.assertEqual(sources[0].text, source.decode())
                self.assertEqual(build_tex_source_from_bytes(payload, "a"), source.decode())

    def test_main_document_expands_relative_includes(self):
        document = build_tex_source([
            TexSource("paper/main.tex", r"\documentclass{article}\begin{document}\input{parts/intro}\end{document}"),
            TexSource("paper/parts/intro.tex", r"\section{Introduction}Included text."),
            TexSource("unrelated.tex", "UNRELATED"),
        ])
        self.assertIn("Included text.", document)
        self.assertNotIn("UNRELATED", document)
        self.assertNotIn(r"\input", document)
        self.assertEqual(build_tex_source([]), "")

    def test_sections_preserve_hierarchy_without_overlapping_text(self):
        sections = extract_sections_from_latex(r"""\documentclass{article}
        \begin{document}\section{Main}First.
        % \section{Ignored}Comment
        \subsection{Detail}Second.\section*{Conclusion}Third.\end{document}Outside""")
        self.assertEqual([part.kind for part in sections], ["section", "subsection", "section"])
        self.assertEqual([part.title for part in sections], ["Main", "Detail", "Conclusion"])
        self.assertEqual([part.text for part in sections], ["First.", "Second.", "Third."])

    def test_cleaning_preserves_prose_math_names_and_link_text(self):
        text = clean_latex_for_embedding(
            r"\textbf{Useful idea}: $x \rightarrow \Omega$ and \href{https://example.com}{a link}."
            "\n% hidden comment\n"
            r"\label{internal-key}Cost is 10\%."
        )
        for phrase in ("Useful idea", "rightarrow", "Omega", "a link", "10%"):
            self.assertIn(phrase, text)
        for removed in ("hidden comment", "internal-key", "https://", "$", r"\textbf"):
            self.assertNotIn(removed, text)

    def test_latin1_sources_are_decoded(self):
        self.assertEqual(list(iter_tex_sources("café".encode("latin-1")))[0].text, "café")


if __name__ == "__main__":
    unittest.main()
