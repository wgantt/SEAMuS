# Resources

Contents:
- `saved_contexts.zip`: files containing the top-k sentences retrieved from the full FAMuS source documents, using the report text as a query. We include contexts retrieved with a couple of dense retrievers, but we obtained our best results with BM25 in the paper (using `k=7`), and so you should use the context files corresponding to this setting if you are trying to reproduce our results.
- `saved_prompts.zip`: contains the exact prompts used to obtain the GPT and Claude results reported in the paper.