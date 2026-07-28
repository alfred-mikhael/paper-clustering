# Work Log — Paper Clustering Project

**Date: July 8, 2026**
**Time spent: 2 hours**
**Goal for this session:** Refactor codebase to be local, and set up systems to improve development efficiency.

## What I worked on

* Initialize git repository for project instead of working on Colab
* Set up worklog.md and readme.md, as well as a project on GPT so everything related to this project is kept in the same place
* Set up miniconda environment for the project
* Improve download from arxiv by:
    * better category detection
    * better failure logging

## Files or data used

List any datasets, papers, scripts, notebooks, or output files.

* data_collection.ipynb
* requirements.txt
* readme.md
* wordklog.md

<!-- ## Results

Write the main outcome.

* Number of papers:
* Number of clusters:
* Useful observations:
* Problems noticed: -->

## Decisions made

Record any choices you made and why.

Decided to skip collecting papers whose primary label is not in the categories list, as they seem to be 
tangentially related at best, and quite irrelevant most of the time.

## Issues or questions

List anything confusing, broken, or worth checking later.

* Might want to refactor failure logging in a cleaner way. Logger class that has an in-memory buffer and 
can flush to a file seems like a better way to handle logging, especially since the construction of the log
message might not be the same in every scenario.

## Next step

Update intro extraction, bulk download Arxiv papers and upload to Supabase, deduplication


**Date: July 9, 2026**
**Time spent: 2 hours**
**Goal for this session:** Update intro extraction, upload papers to supabase and deduplicate. Start logging to files and supabase.

## What I worked on

* Create a better logging system. Now logs things as jsonl
* Make arxiv fetch function more modular and handles errors better

## Files or data used

List any datasets, papers, scripts, notebooks, or output files.

* notebooks/data_collection.ipynb
* src/arxiv_fetch.py
* src/logger.py

<!-- ## Results

Write the main outcome.

* Number of papers:
* Number of clusters:
* Useful observations:
* Problems noticed: -->

## Decisions made

Record any choices you made and why.

Make separate logger class, because I will eventually want to log errors in a database, and I want to reuse the class for logging errors with intro extraction and tex source downloads. Moved functions into a .py file instead of keeping everything in .ipynb so that it is easy to reuse.

## Issues or questions

List anything confusing, broken, or worth checking later.

* Might not want to open and close file every time I log. Could buffer and flush periodically, but that seems overengineered for my use case. 

## Next step

Update intro extraction, bulk download Arxiv papers and upload to Supabase, deduplication

**Date: July 10, 2026**
**Time spent: 3.5 hours**
**Goal for this session:** Update intro extraction, upload papers to supabase and deduplicate. Start logging to supabase. Get preliminary embeddings and mappings. 

## What I worked on

* Used codex to update math extraction into three levels: level 0 where all math is replaced by <MATH> tokens. Level 1 where some common and important expressions are replaced with english hints, and the rest is replaced with <MATH>. Level 2 where all expressions are turned into a normalized english representation, for example "x + y" becomes "x_plus_y", and "$\Pr[x \geq 2] \leq \exp(-2t)" becomes "Pr_x_ge_2_le_exp_neg_2_t". 
* Used codex to optimize the speed of intro extraction, as it was quite slow. I need to review that code more carefully, because I can probably optimize it significantly. Maybe it is best to use a C program for this?
* Logger now logs skipped papers to a file, and can optionally log to stdout
* fetch_arxiv_data now has a (toggleable) progress bar

## Files or data used

List any datasets, papers, scripts, notebooks, or output files.

* notebooks/data_collection.ipynb
* src/arxiv_fetch.py
* src/logger.py
* src/extract_intro.py

<!-- ## Results

Write the main outcome.

* Number of papers:
* Number of clusters:
* Useful observations:
* Problems noticed: -->

## Decisions made

Record any choices you made and why.

Had codex generate different levels of math normalization when doing intro extraction. The first level suprresses all math notation with a <MATH> symbol. The second level replaces some expressions with english hints (i.e. y \in \F_2^n becomes finite field vector), and the third level keeps as much math notation as possible, separated by underscores (i.e. y \in \F_2^n becomes y_in_finite_field_2_n, and \Pr[|x - y| > 10] < 0.1 becomes Pr_abs_x_minus_y_gt_10_lt_0.1). I did this because, while the math notation is of course very important, there is a lot of variation in how people write it, both in symbols and especially in LaTeX macros. For example, the boolean cube {0, 1}^n can be written as 
1. \{0, 1\}^n,  
2. \lbrace 0, 1 \rbrace^n
3. \F_2^n 
4. \bits^n 
5. \Bits^n
6. \F2^n
And so on. Moreover, the same parameters are often given different variable names. Although they are usually chosen from a small range of options, this still adds noise to the model. It will take a lot of experimentation to see the right level of feature extraction that should be done.

## Issues or questions

List anything confusing, broken, or worth checking later.

* Definitely need to review (and probably rewrite) the code generated by Codex. While it is very useful, I do not think it has a place in writing anything except short snippets of boilerplate code. I think the intro extraction from the tex files is probably fine (if perhaps a little slow) but the math cleaning rules need to be verified and rewritten by me. 

## Next step

No need to focus on improving performance at this stage. Before rewriting rules for intro extraction techniques, I should evaluate what I have against a few models. I need to collect a dataset of about 200-300 papers, across a range of research areas and with hard examples for clustering. Then I can run this against a few different embedding models and evaluate the quality of the clusters both qualitatively (manually) and quantitatively (need to resarch what methods are used to evaluate clustering quality)

**Date: July 13, 2026**
**Time spent: 3.5 hours**
**Goal for this session:** Evaluate which level of math extraction and what embedding model is best as a baseline. I need to do this **before** uploading to Supabase. 

## What I worked on

* Evaluated quality of embeddings with different level of math removal and different embeeding models.

## Files or data used

List any datasets, papers, scripts, notebooks, or output files.

* notebooks/first_evals.ipynb

## Results

Write the main outcome.

* SPECTER performs better than BM25, and all-minillm-l12-v2. 
* 'coarse' level of math extraction does (marginally) better than 'fine' and 'no_math', but it is a very small difference
* all-minillm-l12-v2 is not affected at all by the level of math extraction. It performs silghtly worse than SPECTER but not by a lot
* BM25 performs very poorly across all levels of math extraction

## Decisions made

Record any choices you made and why.

* Rather than handpicking papers for the evaluation dataset, I decided to choose a few different categories and use the existing Arxiv keyword search to get the papers. Handpicking 500 papers would take an extremely long time, and it's definitely not worth it for this initial evaluation
* Decided to compare SPECTER and all-minillm-l12-v2. SPECTER has been trained on scientific papers (but not math papers), and all-minillm-l12-v2 is a more general purpose model. It is important to see the performance of both.
* Used codex to generate the code which runs the experiment. I think it was very successful. 

## Issues or questions

List anything confusing, broken, or worth checking later.

* Did not consider the context window size of the models. It probably doesn't make much of a difference whether I just embed title + abstract or title + abstract + intro, since most abstracts will already fill up the entire window, and the first few sentences of an introduction aren't particularly useful. 
* PyTorch is not running on GPU locally and I'm not sure why. It might be better to run the experiments on Colab to save some time, rather than trying to debug why my GPU isn't working.

## Next step

* Test the difference between title + abstract and title + abstract + intro. 
* Test an embedding model which has a large enough context window to fit the entire introduction
* Decide whether to do experiments on Colab or locally? Long term it is better to fix it locally, but it will also be time consuming.

**Date: July 15, 2026**
**Time spent: 2.5 hours**
**Goal for this session:** Continue evaluating which embedding model + level of math removal is most useful. Test whether embedding the introduction is useful at all. If possible, handcraft some better math extraction rules and see about extracting only parts of the intro which are most useful ("Our results", "Proof overview", Theorem statements, etc). 

## What I worked on

* Completed and documented evaluations for embedding models. The experiment was a qualitative evaluation of the top10 nearest neighbours of 2 papers.
* Had Codex simplify the code in extract_intro.py 

## Files or data used

List any datasets, papers, scripts, notebooks, or output files.

* src/extract_intro.py
* notebooks/first_evals.ipynb

## Results

Write the main outcome.

* Fixed PyTorch so it runs on GPU locally. Did this by avoiding the conda installer and using `python -m pip install ...` and installing torch from `--index-url https://download.pytorch.org/whl/cu118`
* Found that title + abstract currently performs the best, so math removal is unnecssary. That's a little disappointing but it's good that I found out now and won't spend much time on it. Hard to pin down a single best model so far. Nomic performed best on the catalytic paper, while SPECTER did best on the LCC paper, while MiniLM did decent in both. BM25 also did very well, and takes extremely little computation time. It might be best to use BM25 as a baseline for experimentation, and test against all the other 3 if it passes the intial filter. 

## Decisions made

Record any choices you made and why.

* Decided to do experiments by qualitatively analyzing the top10 nearest neighbours from a couple of papers. I could've done more papers, but this already took a long time and it's just a preliminary investigation. It's also hard to quantify what makes a paper more relevant than another. 
* Removed the neural network noisy papers from the dataset. These papers just add a bunch of easy examples, which are not so important compared to the hard examples (papers which sound similar but are different).

## Issues or questions


## Next step

* Instead of extracting math from the papers, try to extract a few very important sentences. I will first do this programatically, and then later try with an LLM to do some structured extraction. It will also be helpful to have different embeddings for different purposes, so that it is easier to cluster by techniques and by areas. Most papers have some sort of "proof overview" or "technical overview" or "our techinques" section, but some do not, so it is a challenge to figure out how to extract information about the techniques used in a clear way. 

**Date: July 21, 2026**
**Time spent: 2 hours**
**Goal for this session:** Since math removal is actually not helpful for model performance, try to extract some important sentences from the introduction and use that to enrich the embedding. I definitely want to extract any definitions and theorem statements in the introduction, and maybe in the second 

## What I worked on

Had codex generate some code for extracting enrichment from an introduction. I went back and forth with it a few times about what the regex should be and what sort of information is considered useful. I compared Nomic embeddings title + abstract + enrichment to BM25 with title + abstract + enrichment and BM25 with only title + abstract. Results showed that enrichment did not improve performance at all.  

## Files or data used

List any datasets, papers, scripts, notebooks, or output files.

* notebooks/test_enrichment.ipynb
* src/extract_intro.py

## Results

Write the main outcome.

Enrichment extracts sentences from the introduction that are relevant to the results or the proof techniques used, as well as the theorem statements for the main results. Naively appending the enrichment to the title and abstract did not improve performance for the Nomic embedding, or the BM25 scores. 

## Decisions made

Record any choices you made and why.

* I did not include any mention of prior work in the enrichment, since it would dilute information about the current results and the techniques used. 
* I decided to append the enrichment to the title and abstract, hoping that the Nomic model could learn more about the paper from it

## Issues or questions

* GPT recommends a multi-view embedding, where I try to extract information about techinques separately from information about the area, seperately from the baseline title + abstract, and then combine them all together. That sounds pretty promising, and apparantly it was used for the SPECTER model
* I still need to think about math removal in the enrichment
* I should test an embedding model with __just__ the enrichment, rather than title + abstract + enrichment, to see if the enrichment is really not good or if the issue is too much text diluting the main ideas. 
* I also need to generate a good training data set with some labelled examples so I have a proper testbed. 

## Next step
* Generate a good training data set, with labels, and rerun tests for enrichment using that dataset. 

**Date: July 22, 2026**
**Time spent: 2 hours**
**Goal for this session:** Plan out the training / test data set generation. I need to choose which papers to include, decide how the extraction should work, and make sure that everything will go properly before running it through a teacher model to generate labels. 

## What I worked on

Mostly planned out how I am going to label the data, and simplified enrichment code. 

## Files or data used

List any datasets, papers, scripts, notebooks, or output files.

* src/extract_intro.py

## Results

Write the main outcome.

Here is the plan for generating a labelled dataset: 
1. Must finalize enrichment from introduction
2. Get a dataset of a couple thousand papers across a variety of different fields. I should focus on what is most relevant to me, and adjacent fields. 
3. Select about 75 "anchor" papers which I will measure similarity to. These should be quite influential, and span a wide range of different ideas and fields
4. Get the top10 most related papers from each of BM25, Nomic with enrichment, and SPECTER. Deduplicate and use these as the labelled pairs. There will be a lot of positives, but also a fair amount of semi-related and unrelated papers (hard negatives)
5. Possibly run a model on the enrichment to generate a structured output
6. Use OpenAI batch API to generate labels for each pair of papers. The model will receive the title, abstract, and the enrichment, and should output a score from 0 - 5, as well as a brief explanation of that score (what I ask for depends on how small I want to make the output tokens)
7. This should generate about 1000 labelled pairs. I should manually validate a few hundred (write a python script to help speed this up), and then reserve those for testing. I can use the rest to do some initial fine-tuning. 

## Decisions made

Record any choices you made and why.

* Will probably use the OpenAI batch API on GPT5.6 Luna, since it balances cost and quality. Total cost should be <=$20 for the first 1000 pairs. 
* Will use this knowledge distillation sort of approach because I do not have the time to manually label 15-25k pairs. I can try writing a python script to see roughly how long it takes me. If it takes me roughly 30s per pair to label, then just labelling 1000 pairs will take roughly 8 hours of work - doable, but not pleasant. I can try looking into a semi-supervised approach to make things even cheaper?
* Run a model on the enrichment to generate structured output: this reduces input size, and makes it easier for the teacher model to make good labels allowing me to use a cheaper model. Conversely, it has a fixed cost per new paper, which would be nice to avoid.

## Issues or questions
* How worthwhile is the OpenAI API? What can be done manually versus what should be labelled by a stronger model
* Can I train a small local model to take the enrichment text extracted from the intro by regex and convert it into a more structured format? It will probably be much better in the long run than using an OpenAI model
* Do I really need 15k training examples?

## Next step
Keep experimenting with and finalize enrichment. Get a dataset and choose the anchor papers carefully. Get a small script to see how long it really takes to manually label data, and look into a semi-supervised approach which might be cheaper than using the OpenAI API.  

**Date: July 28, 2026**
**Time spent: 2 hour**
**Goal for this session:** Keep working on text enrichment. Also do better preprocessing on the text before embedding and see if that makes any changes.

## What I worked on
Continued working on the enrichmenet and did research into what sort of clustering people do. I found a good paper on clustering using LLMs, as well as a survey on various clustering techniques and their pros/cons. 

## Files or data used

List any datasets, papers, scripts, notebooks, or output files.

* extract_intro.py

## Results
Enrichment keeps getting better, but it is not complete yet. It still catches some sentences that are not important and misses some important ones.

## Decisions made

Record any choices you made and why.

* I decided NOT to do the text preprocessing which I was planning to do, namely stopword removal and stemming. According to the research I did, this usually reduces the performance of LLMs and text embeddings, since they are trained on full sentences with all stop words, grammer, and transitions. 
* I did put back a little bit of math handling. I just want to replace common constants with their english versions, so that the math doesn't become meaningless. I also removed dollar signs to hopefully reduce noise in the model.

## Issues or questions
* I had this idea of chunking the text of papers so that each chunk fits into the context window of a BERT family embedding. When searching for the top10 nearest neieghbours, I can get the top5 for each cluster as candidates, and then take the top10 candidates which have best average cross-cluster relevance. The issue with this is that it would bias papers which are about the same topic, and use different techniques. 
* What if I use BM25 or TF-IDF to help with the enrichment? I can try by computing BM25 indices for a small dataset, and then using those precomputed indices to give weight to sentences with keywords when doing enrichment. 

## Next step
I really like the idea of using BM25 to help filter in the enrichment step. I will try that out, and I can also learn about and try out some other cool classical techniques I come across, although it's hard to imagine how anything will be much different from BM25 or TF-IDF.