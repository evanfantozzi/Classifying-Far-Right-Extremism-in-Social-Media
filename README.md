# Classifying Far-Right Extremism in Social Media: A Supervised Learning Approach

## Topic

Our model addresses a simple but important question - how likely is a
given English-language social media post to be right-wing extremist?

## Data

- 4670 social media posts, all manually labeled as extremist or not

| **Data Source**               | **Year**    | **\# Extreme** | **\# Non-Extreme** |
|-------------------------------|-------------|----------------|--------------------|
| Stormchaser                   | 2017        | 1500           | 0                  |
| Sentiment140                  | 2009        | 0              | 1500               |
| TwitterAPI (Manually Scraped) | 2024 - 2025 | 51             | 1619               |

## Features

- **TF-IDF**: Weighted term frequency  

- **MiniLM Sentence Embeddings**: Vector-based semantic similarity

- **Profanity:** Presence of "bad" word or not (0 = Not Present, 1 = Present)

- **Subjectivity:** Extracted via TextBlob (0-1 scale where 0 = Objective, 1 = Subjective)

- **Toxicity:** Extracted via Detoxify (0-1 scale where 0 = Not Toxic, 1 = Toxic)

- **Insider Terms:** Presence of insider term from small list (0 = Not Present, 1 = Present)

## Model Selection and Results

We tested **Logistic Regression** and **non-linear Support Vector Machine** models, ultimately choosing the latter due to its higher extremist text recall score, as we are more concerned with avoiding false negatives than false positives.

### Sample Classifications:

| **Text**                                                                                                                                        | **Classification (%)** |
|-------------------------------------------------------------------------------------------------------------------------------------------------|------------------------|
| "I woke up on a white beach with an american flag towel"                                                                                        | Non-Extremist (19%)    |
| "I love the Amazing Race! What a great show about conquering your rivals and finishing number one. I hope they start showing it on Channel USA" | Non-Extremist (23%)    |
| "I don\'t agree with everything that Trump does, but I think he cares about protecting us"                                                      | Non-Extremist (41%)    |
| "We're losing our established identity, and no one seems to care. Guess that's \'progress.\'"                                                   | Extremist (52%)        |
| "I WILL END INFLATION, I WILL STOP THE INVASION OF CRIMINALS COMING INTO OUR COUNTRY, and I WILL BRING BACK THE AMERICAN DREAM"                 | Extremist (68%)        |
| "The communist threat is circling in on our culture"                                                                                            | Extremist (97%)        |

# Running the Model

## Dataset Preparation
See Data.md in the data folder

## Keyword Lists Preparation
See KeywordLists.md in build_models/keyword_lists

Once all files are obtained, from the main directory run: `uv sync`.
This ensures you have the libraries you need. Only need to run once.

## `python3 build_models/create_model_pkls.py`
This learns the model and saves it to a file. Only should be run once.

## `python3 app.py`
This opens the interactive user interface.
