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
**Time spent:  hours**
**Goal for this session:** Evaluate which level of math extraction and what embedding model is best as a baseline. I need to do this **before** uploading to Supabase. 

## What I worked on

* 

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

