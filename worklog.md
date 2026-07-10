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
**Time spent: 2 hours**
**Goal for this session:** Update intro extraction, upload papers to supabase and deduplicate. Start logging to supabase. Get preliminary embeddings and mappings. 

## What I worked on

* Used codex to update math extraction into three levels: level 0 where all math is replaced by <MATH> tokens. Level 1 where some common and important expressions are replaced with english hints, and the rest is replaced with <MATH>. Level 2 where all expressions are turned into a normalized english representation, for example "x + y" becomes "x_plus_y", and "$\Pr[x \geq 2] \leq \exp(-2t)" becomes "Pr_x_ge_2_le_exp_neg_2_t". 

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
