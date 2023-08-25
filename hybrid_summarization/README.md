# hybrid_summarization
Evaluating hybrid summarization methods using metrics such as SMART, SummaC, and others as suggested.


## SMART Scores

The folder in the repo smart_eval is the code from the SMART paper, but I had to do some minor tweaks in the code to make it run. I believe there is some version mismatch ocurring that causes the type of the score_matrix to be a chrf (?) object instead of floats. I cast the elements of the numpy arrays to the right type and it fixed the problem and passed the scorer_text.py file. All changes are in scorer.py, and begin on the following lines: 91, 183, 234. There may be a better way to fix this, but I prioritized passing the tests and repeated code when necessary.

Metric_compute.ipynb includes code for running all ROUGE, SMART, and SummaC examples. Make sure to use a kernel with Python 3.8.