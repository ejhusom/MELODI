#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extracting features from text.

Author:
    Erik Johannes Husom

Created:
    2024-03-20

"""
import string
import pandas as pd
import numpy as np
import spacy
from textblob import TextBlob
import textstat
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# Load the spaCy English model
nlp = spacy.load("en_core_web_sm")

# Ensure NLTK stop words are downloaded
import nltk
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

def extract_features(df):
    # Ensure the text data is in string format
    df["prompt"] = df["prompt"].astype(str)

    # # Initialize columns for the features
    # df["word_count"] = 0
    # df["sentence_count"] = 0
    # df["avg_word_length"] = 0
    # df["named_entity_count"] = 0
    # df["noun_count"] = 0
    # df["verb_count"] = 0
    # df["adj_count"] = 0
    # df["sentiment_polarity"] = 0
    # df["sentiment_subjectivity"] = 0
    # df["flesch_reading_ease"] = 0

    for index, row in df.iterrows():
        prompt = row["prompt"]

        # Tokenize the prompt into sentences and words
        sentences = sent_tokenize(prompt)
        words = word_tokenize(prompt)
        unique_words = set(words)

        # Compute basic counts
        df.at[index, "word_count"] = len(words)
        df.at[index, "sentence_count"] = len(sentences)
        df.at[index, "avg_word_length"] = sum(len(word) for word in words) / len(words)

        # Additional basic features
        df.at[index, "word_diversity"] = len(unique_words) / len(words) if words else 0
        df.at[index, "unique_word_count"] = len(unique_words)
        df.at[index, "avg_sentence_length"] = len(words) / len(sentences) if sentences else 0
        df.at[index, "punctuation_count"] = sum(prompt.count(w) for w in string.punctuation)
        df.at[index, "stop_word_count"] = sum(1 for word in words if word.lower() in stop_words)
        df.at[index, "long_word_count"] = sum(1 for word in words if len(word) > 6)

        # Named Entity Recognition and POS tagging with spaCy
        doc = nlp(prompt)
        df.at[index, "named_entity_count"] = len(doc.ents)
        pos_counts = doc.count_by(spacy.attrs.POS)
        df.at[index, "noun_count"] = pos_counts.get(spacy.symbols.NOUN, 0)
        df.at[index, "verb_count"] = pos_counts.get(spacy.symbols.VERB, 0)
        df.at[index, "adj_count"] = pos_counts.get(spacy.symbols.ADJ, 0)
        adverb_count = sum(1 for token in doc if token.pos_ == "ADV")
        pronoun_count = sum(1 for token in doc if token.pos_ == "PRON")
        df.at[index, "adverb_count"] = adverb_count
        df.at[index, "pronoun_count"] = pronoun_count
        df.at[index, "prop_adverbs"] = adverb_count / len(words) if words else 0
        df.at[index, "prop_pronouns"] = pronoun_count / len(words) if words else 0

        # Sentiment Analysis with TextBlob
        blob = TextBlob(prompt)
        df.at[index, "sentiment_polarity"] = blob.sentiment.polarity
        df.at[index, "sentiment_subjectivity"] = blob.sentiment.subjectivity

        # Readability Scores with textstat
        df.at[index, "flesch_reading_ease"] = textstat.flesch_reading_ease(prompt)
        df.at[index, "flesch_kincaid_grade"] = textstat.flesch_kincaid_grade(prompt)
        df.at[index, "gunning_fog"] = textstat.gunning_fog(prompt)
        df.at[index, "smog_index"] = textstat.smog_index(prompt)
        df.at[index, "automated_readability_index"] = textstat.automated_readability_index(prompt)
        df.at[index, "coleman_liau_index"] = textstat.coleman_liau_index(prompt)
        df.at[index, "linsear_write_formula"] = textstat.linsear_write_formula(prompt)
        df.at[index, "dale_chall_readability_score"] = textstat.dale_chall_readability_score(prompt)
        df.at[index, "text_standard"] = textstat.text_standard(prompt)
        df.at[index, "spache_readability"] = textstat.spache_readability(prompt)
        df.at[index, "mcalpine_eflaw"] = textstat.mcalpine_eflaw(prompt)
        df.at[index, "reading_time"] = textstat.reading_time(prompt)
        df.at[index, "fernandez_huerta"] = textstat.fernandez_huerta(prompt)
        df.at[index, "szigriszt_pazos"] = textstat.szigriszt_pazos(prompt)
        df.at[index, "gutierrez_polini"] = textstat.gutierrez_polini(prompt)
        df.at[index, "crawford"] = textstat.crawford(prompt)
        df.at[index, "osman"] = textstat.osman(prompt)
        df.at[index, "gulpease_index"] = textstat.gulpease_index(prompt)
        df.at[index, "wiener_sachtextformel"] = textstat.wiener_sachtextformel(prompt, variant=1)  # Note: variant parameter is required for wiener_sachtextformel
        df.at[index, "syllable_count"] = textstat.syllable_count(prompt)
        df.at[index, "lexicon_count"] = textstat.lexicon_count(prompt)
        df.at[index, "sentence_count"] = textstat.sentence_count(prompt)
        df.at[index, "char_count"] = textstat.char_count(prompt)
        df.at[index, "letter_count"] = textstat.letter_count(prompt)
        df.at[index, "polysyllabcount"] = textstat.polysyllabcount(prompt)
        df.at[index, "monosyllabcount"] = textstat.monosyllabcount(prompt)


        # Task Difficulty Indicators
        question_marks = prompt.count("?")
        exclamation_marks = prompt.count("!")
        # Number of question or exclamatory sentences might indicate task difficulty
        df.at[index, "question_marks"] = question_marks
        df.at[index, "exclamation_marks"] = exclamation_marks

        # Semantic Depth and Novelty
        # Using sentence embedding variance as a proxy for semantic depth/novelty
        # Higher variance might indicate more diverse or complex ideas
        sentence_embeddings = [nlp(sentence.text).vector for sentence in doc.sents]
        if sentence_embeddings:
            variance = np.var(sentence_embeddings, axis=0).mean()
        else:
            variance = 0
        df.at[index, "sentence_embedding_variance"] = variance

        # Interactivity and Context Dependence
        # Approximated by checking for references to previous discourse or external contexts
        personal_pronouns = sum(token.text.lower() in ["i", "we", "you", "he", "she", "they"] for token in doc)
        df.at[index, "personal_pronouns"] = personal_pronouns

        # Specific Knowledge or Data Requirement
        # Proxy by counting named entities; more named entities might indicate more specific knowledge requirements
        named_entities = len(doc.ents)
        df.at[index, "named_entities"] = named_entities

        # Creativity Requirement
        # This is challenging to quantify directly without deep semantic analysis,
        # but we can use indirect indicators like the presence of adjectives and adverbs,
        # assuming more descriptive language might be used in creative tasks
        adjectives = sum(token.pos_ == "ADJ" for token in doc)
        adverbs = sum(token.pos_ == "ADV" for token in doc)
        df.at[index, "adjectives"] = adjectives
        df.at[index, "adverbs"] = adverbs

        # Complexity Through Readability Scores (can be directly added from textstat)
        df.at[index, "coleman_liau_index"] = textstat.coleman_liau_index(prompt)
        df.at[index, "dale_chall_readability_score"] = textstat.dale_chall_readability_score(prompt)

        # Interaction between length and complexity
        df["length_x_complexity"] = df["word_count"] * df["sentence_embedding_variance"]

        # Interaction between question marks and named entities
        # Hypothesizing that questions about specific entities might be more complex
        df["questions_about_entities"] = df["question_marks"] * df["named_entities"]

        # Ratio of adjectives and adverbs to total word count - might indicate descriptive complexity
        df["desc_complexity_ratio"] = (df["adjectives"] + df["adverbs"]) / df["word_count"]

        # Squared word count - to capture non-linear effects of prompt length on energy consumption
        df["word_count_squared"] = df["word_count"] ** 2

        # Cubed average sentence length - might highlight the non-linear impact of very long sentences
        df["avg_sentence_length_cubed"] = df["avg_sentence_length"] ** 3

        # Compute the lexical diversity (unique words / total words)
        df["lexical_diversity"] = df["unique_word_count"] / df["word_count"]

        # # Count of modal verbs - might indicate speculative or conditional statements that are complex to process
        # df["modal_verbs_count"] = df["prompt"].apply(lambda text: sum(token.text.lower() in ["can", "could", "may", "might", "shall", "should", "will", "would", "must"] for token in nlp(text)))

        # # Complexity of questions - hypothesizing that more words in questions indicate deeper queries
        # df["question_complexity"] = df["prompt"].apply(lambda text: np.mean([len(sent.text.split()) for sent in nlp(text).sents if "?" in sent.text]) if any("?" in sent.text for sent in nlp(text).sents) else 0)

    return df

# df = extract_features(df)




