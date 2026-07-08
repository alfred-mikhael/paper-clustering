# Automatically up-to-date literature library
This project automatically scrapes the latest papers in TCS and Combinatorics from Arxiv, groups them by technique and area, and makes them avaialble to the user via similarity search or metadata search. This is exactly what I used to spend 30 minutes each morning doing, but now it is automated! 

## Features
1. Store latest Arxiv papers, and make them available via title search, author search, or keyword search.
2. Cluster papers by area and by technique used and allow for a semantic similarity search. 
3. Visualize papers to visualize clusters of related papers, either by area or technique.
4. (Optional) Draw a citation graph between papers, and make that available 
5. (Optional) AI Digest of paper

## Pipeline 
`data-collection.ipynb` downloads data from Arxiv, extracts metadata and does some simple cleaning, and uploads it to Supabase, as well as logging any failures
TODO: Add the rest as you go

## Models used

## Evaluations

## Demo

