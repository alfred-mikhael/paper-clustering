"""Extract important passages from a paper using a fine-tuned SciBERT classifier."""

from paper_clustering.data_models import Paper, TechniqueInfo
from paper_clustering.technique_classifier import TechniqueClassifier


def extract_techniques(paper: Paper, classifier: TechniqueClassifier) -> TechniqueInfo:
    # 1. Run SciBERT model on each paragraph
    # 2. Retrieve top 10 paragraphs
    # 3. If possible, merge adjacent paragraphs which discuss the same point
    # 4. Choose the best 3 paragraphs with a diversity metric
    pass
