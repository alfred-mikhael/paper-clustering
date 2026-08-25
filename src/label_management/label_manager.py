"""API for generating weak labels for paragraph excerpts of papers, and keeping track of human labelled ones"""

from .regex_labels import label_paper
from .slm_labels import SLMWeakLabelGen
from ..extract_text import get_sections
from ...scripts.label_sentences import main
import requests
from dataclasses import dataclass


@dataclass(frozen=False)
class Label:
    val: float
    human: bool
    paper_id: str
    section_id: int
    paragraph_id: int


class LabelManager:
    def __init__(self, server_url=None, request_timeout=300):
        self.session = requests.Session()
        self.slm_generator = SLMWeakLabelGen(server_url, request_timeout)
        # keys are arxiv_ids, values are lists of Labels
        self.human_labels = dict()
        self.weak_labels = dict()
        self.predicted = dict()

    def generate_weak_labels(self, arxiv_id, force=False):
        # Don't accidentally rewrite the labels unless forced.
        if not force and arxiv_id in self.weak_labels:
            return
        sections = get_sections(self.session, arxiv_id)
        slm_labels = self.slm_generator.label_paper(sections)
        regex_labels = label_paper(sections)

        i = 0
        for section_id in range(len(sections)):
            for paragraph_id in range(len(sections[section_id])):
                self.weak_labels[arxiv_id].append(
                    Label(
                        val=0.7 * slm_labels[i] + 0.3 * regex_labels[i],
                        human=False,
                        paper_id=arxiv_id,
                        section_id=section_id,
                        paragraph_id=paragraph_id,
                    )
                )
                i += 1
        return self.weak_labels[arxiv_id]

    def generate_human_labels(self, arxiv_id, force=False):
        if not force and arxiv_id in self.weak_labels:
            return
        main(...)

    def find_max_entropy_labels(): ...
