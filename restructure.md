# Plan for narrowing the project's scope
## Summary
Shift from general proof-technique similarity to a catalogue of documented connections between fields, alongside the existing topic visualization. The visualization product already exists in a separate repository; this repository supplies ingestion, embeddings, clustering, storage, and retrieval infrastructure.

The next research priority is to establish whether a teacher can identify these connections and extract useful, evidence-supported explanations of why the ideas were applied. Compare context requirements and test whether these explanations improve retrieval before choosing a student architecture or indexing at scale. Historical novelty is not required: a well-known connection can still be new and useful to a user.

## Current state of project
Currently, the project's goal is "given a paper, find the $n$ papers which use the most similar proof techniques", where $n$ is configurable. This is accomplished by semantic search on selected passages from papers. The hypothesis is, since (mathematical) papers already describe their techniques in some detail, appropriately selected passages from papers which describe similar techniques will have embeddings which are close together. In order to handle the scale of math and CS papers (roughtly 1.5 million and growing), passage selection must be done efficiently. In particular, frontier models are prohibitively expensive for the purpose of selecting which passages are most useful. Therefore, passage selection is done by distilling an LLM teacher into a much smaller SciBERT classifier. 
## Motivation: why the change
The project makes the following assumptions, either implicity or explicitly:
1. Passages describing similar techniques will embed nearby in space, regardless of notational differences
2. Authors will describe the techniques used in their main contributions clearly and succinctly
3. There are stand-alone paragraphs in a paper which clearly describe the proof technique, without needing context from the rest of the paper
4. There is an implicit human understanding of "similar" techniques. That is, a human expert can reliably identify when two techniques are similar or not.

The reason for the change is that not all of these assumptions are satisifed. 

### Assumption 1
There is some evidence that this assumption is not satisfied. For example, look at the notebook `notebooks/technique_clustering.ipynb`. The dataset used in this notebook contains paper-pairs which specifically use similar techniques, along with some added noise. Even setting the minimum cluster size very low, we see that the clusters are not very high quality. Clusters end up being mostly by area rather than containing cross-field matches like desired. Of course, this could be an issue with selection as well, and more evidence should be gathered to refute or support this hypothesis.

Poor clustering does not by itself establish poor ranked retrieval. Run a bounded diagnostic using manually selected technique passages, known matching papers, and same-topic distractors. Measure the ranks of the known matches to separate passage-selection problems from representation problems. The ambiguity in assumption 4 remains an independent motivation for the scope change.

### Assumption 2
The initial scope can focus on explicitly described connections and accept incomplete coverage of papers that do not state them clearly. This is a coverage decision, not an assumption that all authors write this way or that excluded papers have little research value.

### Assumption 3
This assumption is often not satisfied. In many papers, the description of the techniques used spans more than one or two paragraphs, sometimes reaching a few pages in length. Nonetheless, these papers might be very useful for a human reader. In fact, as the field becomes more well-established and technically sophisticated, these sorts of papers become increasingly more common. I will note that this can be handled by feeding the entire "proof overview" section into a transformer-based classifier, but this is too expensive for scaling to >1M papers (at least with the resources I have access to).

### Assumption 4
Assumption 4 is a glaring violation. What seems like a similar technique to one expert might be a distinct technique to another. The perceived similarity depends on the reader's background knowledge, and more centrally the level of abstraction at which the reader is thinking. Additionally, two ideas can be similar in some ways but not others. 

For example, should one consider a proof which uses the lovasz local lemma and a proof which uses entropy compression to be similar? For an existence proof, with high probability (pun intended), the two arguments are interchangeable and so they are quite similar in that sense. However, for a paper more focused on the algorithmic aspect, the LLL cannot be used in place of entropy compression. Another example, should one consider the second moment method to be similar to the lovasz local lemma? Depending on the level of abstraction, you can consider them both probabilistic methods (therefore similar), or you can recognize that they are usually used in quite different applications (and so different). To somebody who has limited exposure to probabilistic methods, these two arguments can seem quite similar, but to an expert in the field they are very different. This raises the question: what is the right level of abstraction for defining similarity in this project? If we look at a very low level, then we will not find any similar techniques because each application is slightly different, however if we look at too high of a level the results are not satisfying nor useful to experts. Perhaps more importantly, we don't have control over the level of abstraction used in papers. The "proof overview" (or similar) section of a paper uses a certain level of abstraction which differs from one author to the next, and it is not clear how to compare even the same technique written at two different levels of abstraction. 

One idea is to have retrieval operate at multiple levels of abstraction. "proof overview" sections often start with a very high level overview and go into more detail step-by-step. This makes it possible, potentially, to have retrieval occur in a hierarchical way. However, there are some issues that would need to be addressed: 
* Do enough papers do this to have a meaningful amount of data at each level of abstraction? Not all papers are written in this way. How should a paper with only a low-level summary be dealt with?
* Can we reliably detect different levels of abstraction? It is not clear how to define or recognize levels of abstraction.

## Proposed new scope
The new scope encompasses two related goals:

1. Improve the existing interactive visualization of math/CS fields: hierarchical labels, paper/topic search and navigation, and field filtering, starting with arXiv categories. Frontend changes belong in the separate visualization repository.
2. Build a catalogue of documented connections between fields. If useful purposes can be extracted, search for connections that helped authors solve similar problems. The goal is knowledge sharing, not diagnosing what blocks a user's research or prescribing the correct next step.

### What counts as a connection

Here, "reduction" is used broadly to include applications of ideas that are not common in the receiving field, including importing a theorem or method without a formal reduction between problems. For example, applying Shearer's lemma in database theory is within scope. The annotation rubric should distinguish formal reductions, reformulations, and applications of ideas from other areas without excluding the latter.

The working hypothesis is that, at least when an idea is first imported, authors explicitly describe its origin elsewhere. Start by detecting these explicit descriptions. Whether an idea is uncommon depends on field and time; a missing origin statement does not prove that no transfer occurred. Later papers may omit the origin, and the original paper may be absent from the corpus. Measure and document these coverage limits rather than treating them as negatives by definition.

Include both current contributions and prior-work descriptions. There is no requirement to determine historical priority: novelty to the user is what matters. Preserve the source paper and supporting passages, and consolidate repeated descriptions when presenting results.

### Context and purpose are experimental questions

Identifying a connection may require less detail than understanding a proof technique. The hypothesis is that many novel reductions are clearly outlined in the abstract or introduction, without requiring their technical construction. Test this by comparing isolated paragraphs, abstract/introduction context, and whole-paper context on the same papers.

A teacher using whole-paper context may identify a supporting paragraph that cannot be classified correctly in isolation. Record this distinction before training a paragraph classifier. If the task needs context beyond a BERT-style classifier's window, investigate a different efficient approach rather than assuming distillation removes that requirement.

Extracting "why" is the more pressing uncertainty for search. A useful purpose should describe the authors' objective and what the imported idea enables, and, when stated, the obstacle it addresses. Generic motivations such as "obtain a stronger bound" may be accurate but unhelpful for cross-field retrieval. Missing explanations must remain missing rather than being supplied from the teacher's outside knowledge.

The catalogue and visualization have value independently. Purpose-based search is conditional on demonstrating that supported explanations can be extracted and improve discovery. Keep topic-based map positions separate from connection links; reduction-linked papers need not be moved closer together in the topic embedding.

## Actionable steps

### Goal 1: improve the existing visualization

1. Define concrete navigation tasks for evaluating label display, paper/topic search, and category filtering. Identify the data/API changes needed here and the UI changes needed in the visualization repository.
2. Verify that category metadata survives ingestion, local exports, database storage, and the interface consumed by the map. Add missing support where needed.
3. Update label display and filtering in the visualization repository, then expand the indexed corpus using downloaded metadata. Measure memory, indexing time, and map responsiveness on a representative subset before scaling.
4. Evaluate contrastive fine-tuning for topic visualization against the current embeddings and navigation tasks. Retain it only if it improves the map; do not use cross-field connections as automatic topic-similarity labels.
5. Optionally overlay evidence-backed connections on the topic map or create a separate topic-connection view once extraction quality is established.

### Goal 2: validate connection and purpose extraction before scaling

1. **Define the rubric and revise the teacher prompt.** Include positive, negative, and borderline examples of formal reductions, reformulations, and imported theorem/method applications. Mere co-occurrence, generic method use, or switching methods without evidence of a qualifying connection is insufficient. Specify the evidence needed to classify a transfer as cross-field; retain uncertain cases explicitly.
2. **Build a human-reviewed development set of around 100 papers.** Include same-field reductions, cross-field transfers, papers with no qualifying connections, hard negatives, and examples whose interpretation requires surrounding context. Annotate supporting passages and distinguish paper-level absence from insufficient paragraph context. Keep extraction failures separate from negative examples.
3. **Reserve an independent final benchmark.** Use a larger set with realistic class prevalence, including fewer cross-field positives than in the curated development set. Keep papers and closely related versions out of both sets, and avoid tuning prompts on the final benchmark. Define acceptance criteria before examining final results.
4. **Compare context conditions on the same development papers.** Evaluate paragraph-only, abstract/introduction, and whole-paper teacher inputs. Measure connection precision and recall, evidence support, and context requirements. Record truncation, source extraction failures, latency, and token usage. For each identified passage, assess whether a student seeing only that passage could make the same decision.
5. **Test evidence-supported purpose extraction.** Try separate fields for the authors' objective, explicitly stated obstacle, and imported idea's role or benefit, each with supporting text. Allow missing fields. Review factual support, specificity, and usefulness separately; a valid JSON record and an exact quote do not establish that the interpretation is correct. Evaluate connection detection and purpose extraction independently.
6. **Test whether purposes improve retrieval.** On a small manually checked catalogue, compare purpose-based search with existing passage retrieval. Use queries seeking reductions or transfers that solved similar problems, including across different terminology and fields. Judge usefulness of the top results and retain evidence for inspection. This experiment does not ask the system to diagnose a user's research problem. If purposes do not improve discovery, keep the catalogue/map work separate and revisit search before scaling it.
7. **Choose an efficient architecture from the results.** Distill a paragraph classifier only if the context experiment supports that input. Otherwise evaluate context-aware alternatives. A candidate classifier does not itself generate structured purposes: specify the downstream extraction stage and measure the total cost, including missed candidates and any remaining teacher calls. Use active learning where it addresses observed errors.
8. **Benchmark the complete pipeline before corpus-wide indexing.** On the held-out set, report detection, purpose faithfulness, retrieval usefulness, error rates, and inference cost. Validate the student and any extraction stage together. Proceed only when the predeclared quality and resource criteria are met.
9. **Index and integrate validated outputs.** Store connection passages, purposes when supported, paper provenance, and model/prompt versions. Reuse the existing embedding and retrieval infrastructure where suitable, specifying which fields are embedded and how repeated descriptions are grouped. Integrate connections with the existing visualization while preserving topic-based positions.

In parallel, run the bounded original-technique retrieval diagnostic described under assumption 1. It informs the interpretation of earlier results without making rescue of the old objective a prerequisite for this plan.
