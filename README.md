This Program aims to cluster assignee names into clusters with other patents belonging to the same company. A combination of fuzzy string matching and NLTK POS tagger is used to find the similarity between the assignee names and cluster them based on the fuzzy similarity. The data is processed and cleaned and run through 2 stages of clustering. 

**Functionality Breakdown**

The program performs the following key actions: 

**Data Preprocessing:** 

Reads company names from a text file (user-specified path). 

Splits each line into patent number (optional) and assignee name. 

Handles cases with multiple assignees separated by semicolons. 

Cleans the assignee names by: 

Converting them to uppercase. 

Removing stopwords commonly used in company suffixes and prefixes (e.g., "Inc.", "LLC", "GmbH"). 

Removing special characters and extra whitespace. 

**Fuzzy Matching:** 

Implements a custom similarity function that focuses on non-overlapping words between company names. This helps identify similar names with variations in stopwords or order of words. 

Clusters company names based on a user-defined fuzzy matching threshold (default: 95%). Names exceeding the threshold are considered similar. 

**Part-of-Speech (POS) Tagging:**

Analyzes single-word company names using NLTK's POS tagger. 

Focuses on words tagged as "NNP" (proper nouns) or "NN" (common nouns) as potential cluster names. 

This step refines clustering by ensuring single-word names are grouped with similar proper nouns. 

Usage 

Prerequisites: 

Python 3.x 

**Libraries:** 

fuzzywuzzy (pip install fuzzywuzzy) 

nltk (pip install nltk) using nltk.download('punkt') 

re  

**Steps: **

Clone or download this repository. 

Update the file path in the with open statement within the script (company_name_normalization.py) to point to your data file containing company names (one company name per line, optionally with a tab-separated patent number). 

Install the required libraries using pip install fuzzywuzzy nltk. 

Run the script using python company_name_normalization.py. 

**Output:**

The program generates a file named "cnn_op.txt" containing the clustered company names. Each line represents a cluster of similar company names, potentially including the original patent number