# Data

This project uses **MovieLens latest-small** (`ml-latest-small`) from GroupLens Research, University of Minnesota.

- Page: https://grouplens.org/datasets/movielens/latest/
- Direct download: https://files.grouplens.org/datasets/movielens/ml-latest-small.zip

Unzip it and put `ratings.csv`, `movies.csv`, `tags.csv` and `links.csv` in this `data/` folder.
If the files are missing, the first code cell of Section 2 in `notebook.ipynb` downloads and unpacks them automatically.

The raw CSV files are **not** committed to the repository (see `.gitignore`).

**Licence:** GroupLens usage licence (in the dataset's README.txt). Research and educational use only, no commercial use without permission, no implied endorsement, and you must cite:
F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM TiiS 5(4): 19:1-19:19. https://doi.org/10.1145/2827872
