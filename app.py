import string
import numpy as np
import pandas as pd
import streamlit as st
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import contractions
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# PART 1: DATA CLEANING

# Function to load dataset
def load_data(filename):
    try:
        df = pd.read_csv(filename)
        print(f"Dataset shape: {df.shape}")
        print(df.sample(5, random_state=42))
        return df
    except FileNotFoundError:
        print(f"Error: File not found.")
        return None

# Function to rename columns
def rename_columns(df, new_column_mapping):
    df = df.rename(columns=new_column_mapping)
    print(f"Setting new column names: {df.columns} \n")
    return df

def drop_unnecessary_columns(df, unnecessary_columns):
    print(f"Columns dropped: {unnecessary_columns} \n")
    return df.drop(columns=unnecessary_columns)

# Function to handle missing values
def handle_missing_values(df):
    missing_values_per_column = df.isnull().sum() # Find missing value count per column
    print(f"Number of missing values per column: \n {missing_values_per_column} \n")
    print(f"Total number of rows in the dataset: {len(df)}")
    df = df.dropna() # Drop rows with missing values
    print(f"Dropping all rows with missing values... Done.")
    print(f"Number of complete rows in the dataset: {len(df)} \n")
    return df

# Function to convert selected columns to lowercase
def convert_to_lowercase(df, columns_to_convert):
    for col in columns_to_convert:
        df[col] = df[col].str.lower()
    print(f"Converted columns {columns_to_convert} to lowercase.")
    return df

# Function to remove punctuation from a string
def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

# Function to apply punctuation removal to necessary columns
def apply_punctuation_removal(df, columns_to_remove_punctuation_from):
    for col in columns_to_remove_punctuation_from:
        df[col] = df[col].apply(remove_punctuation)
    print(f"Removed punctuation from columns {columns_to_remove_punctuation_from}.")
    return df

# Load stopwords
nltk.download("stopwords")
stop_words = set(stopwords.words('english'))
nltk.download("punkt_tab")

# Function to remove stopwords
def remove_stopwords(text):
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return " ".join(filtered_words)

# Function to apply stopword removal to necessary columns
def apply_stopword_removal(df, columns_to_remove_stopwords_from):
    for col in columns_to_remove_stopwords_from:
        df[col] = df[col].apply(remove_stopwords)
    print(f"Removed stopwords from columns {columns_to_remove_stopwords_from}.")
    return df

# Function to expand contractions
def expand_contractions(text):
  return contractions.fix(text)

# Function to apply contraction expansion to necessary columns
def apply_contraction_expansion(df, columns_to_expand_contractions):
    for col in columns_to_expand_contractions:
        df[col] = df[col].apply(expand_contractions)
    print(f"Expanded contractions in columns {columns_to_expand_contractions}")
    return df

# Load WordNet
nltk.download("wordnet")

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Function to lemmatize necessary columns
def lemmatize_text(df, columns_to_lemmatize):
    for col in columns_to_lemmatize:
        df["review_text"] = df["review_text"].astype(str).apply(
            lambda x: " ".join([lemmatizer.lemmatize(word) for word in x.split()]))
    print(f"Lemmatized text in columns {columns_to_lemmatize}.")
    return df

# Function combining all data cleaning steps into one package
def clean_data(df):
    new_column_mapping = {"Id": "row_id", "ProductId": "product_id", "UserId": "user_id", "ProfileName": "profile_name", 
                      "HelpfulnessNumerator": "helpfulness_numerator", "HelpfulnessDenominator": "helpfulness_denominator",
                      "Score": "score", "Time": "time", "Summary": "review_title", "Text": "review_text"}
    unnecessary_columns = {"row_id", "user_id", "profile_name", "time"}
    columns_to_convert_to_lowercase = ["review_title", "review_text"]
    columns_to_remove_punctuation_from = ["review_title", "review_text"]
    columns_to_remove_stopwords_from = ["review_title", "review_text"]
    columns_to_expand_contractions = ["review_title", "review_text"]
    columns_to_lemmatize = ["review_title", "review_text"]
    df = rename_columns(df, new_column_mapping)
    df = drop_unnecessary_columns(df, unnecessary_columns)
    df = df.drop_duplicates()
    df = handle_missing_values(df)
    df = convert_to_lowercase(df, columns_to_convert_to_lowercase)   
    df = apply_punctuation_removal(df, columns_to_remove_punctuation_from)
    df = apply_stopword_removal(df, columns_to_remove_stopwords_from)
    df = apply_contraction_expansion(df, columns_to_expand_contractions)
    df = lemmatize_text(df, columns_to_lemmatize)
    print(f"\nData has been fully cleaned and is ready for analysis.\n")
    return df

# Function to load and clean data
@st.cache_data
def load_and_clean_data():
    df = load_data("Food Reviews 10k.csv") # Test performance with different numbers of entries
    df = clean_data(df)
    return df

df = load_and_clean_data()

# Print out example product IDs to test with
print(df["product_id"].value_counts().head(3))

# Function to assign positive, neutral or negative sentiment to a review
def assign_sentiment(score):
    if score >= 4:
        return 1 # Positive
    elif score == 3:
        return 0 # Neutral
    else:
        return -1 # Negative

# Add sentiment column to the dataset
df["sentiment"] = df["score"].apply(assign_sentiment)

# Create TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000)

# Select features and label
x = df["review_text"]
y = df["sentiment"]

# Perform train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Fit and transform training and testing data
x_tfidf = tfidf_vectorizer.fit_transform(df['review_text']) 
x_train_tfidf = tfidf_vectorizer.fit_transform(x_train)
x_test_tfidf = tfidf_vectorizer.transform(x_test)

# PART 2: INDIVIDUAL PRODUCT ANALYSIS
# ___________________________________________________________________________________________

# Train LDA model for sentiment analysis
lda_model = LatentDirichletAllocation(n_components=5, random_state=42)
lda_model.fit(x_train_tfidf)

# Function that returns key statistics for a given product
def get_product_stats(product_id):

    # Check if the product ID input is valid
    try:
        product_df = df[df['product_id'] == product_id]
        if product_df.empty:  # Check if the DataFrame is empty (no matching product ID)
            print(f"Error: Product ID '{product_id}' not found in the dataset.")
            return None  # Return None to indicate failure
    except KeyError:  # Catches true KeyErrors if the column doesn't exist
        print(f"Error: Product ID column not found.") # More appropriate error message
        return None

    # Calculate key metrics
    average_sentiment = product_df['sentiment'].mean()
    average_star_rating = product_df['score'].mean()
    review_count = len(product_df)
    
    # Calculate percentage of 5-star reviews
    num_five_star_reviews = len(product_df[product_df['score'] == 5])
    percentage_five_star = (num_five_star_reviews / review_count) * 100
    
    # Topic Modeling
    product_reviews_tfidf = tfidf_vectorizer.transform(product_df['review_text'])
    topic_distributions = lda_model.transform(product_reviews_tfidf)
    dominant_topics = topic_distributions.argmax(axis=1)

    # Group reviews by dominant topics
    reviews_by_topic = {}
    for i, topic in enumerate(dominant_topics):
        if topic not in reviews_by_topic:
            reviews_by_topic[topic] = []
        reviews_by_topic[topic].append(product_df['review_text'].iloc[i])  # Add the actual review text

    top_words_per_topic = {}  # Store top words for each topic for this product

    # Iterate through the topics found for this product
    for topic_id, reviews in reviews_by_topic.items():
        # Recalculate TF-IDF only for this product's reviews related to this topic
        topic_reviews_tfidf = tfidf_vectorizer.transform(reviews)

        # Calculate the average TF-IDF score for each word for this topic
        word_scores = np.asarray(topic_reviews_tfidf.mean(axis=0)).flatten()

        # Get the indices of the top 5 words
        top_word_indices = word_scores.argsort()[-5:][::-1] # Get top 5, reverse order

        # Get the actual top words
        top_words = [tfidf_vectorizer.get_feature_names_out()[i] for i in top_word_indices]
        top_words_per_topic[f"Topic {topic_id + 1}"] = top_words  # Store the top words

    return {
        "product_id": product_id,
        "average_sentiment": average_sentiment,
        "average_star_rating": average_star_rating,
        "review_count": review_count,
        "percentage_five_star": percentage_five_star,
        "key_themes": top_words_per_topic
    }

# Function to get all reviews for a selected product
def get_product_reviews(product_id):
    product_reviews = df[df["product_id"] == product_id]
    if product_reviews.empty:
        print(f"No reviews found for product ID {product_id}.")
        return None
    return product_reviews

# Function to output product stats in an organized way
def print_product_stats(product_id):
    product_stats = get_product_stats(product_id)
    if product_stats:
        print(f"**Statistics for Product ID {product_id}:**") 
        print(f"Number of Reviews: {product_stats['review_count']}") 
        print(f"Average Sentiment: {product_stats['average_sentiment']:.2f}") 
        print(f"Average Star Rating: {product_stats['average_star_rating']:.2f}")
        print(f"Percentage of 5-Star Reviews: {product_stats['percentage_five_star']:.1f}% \n")
        print(f"Key Themes: {product_stats['key_themes']}")
    else:
        print("No data available for the selected product.")    

# PART 3: STREAMLIT APP
# ___________________________________________________________________________________________

import streamlit as st

# --- Streamlit Layout ---
st.title("Amazon Customer Sentiment Analysis Tool (ACSAT)")
st.write("Explore product reviews and gain insights.")

# --- Sidebar Controls ---
st.sidebar.header("Product Selection")

# Get product ID from user
test_product_id = "B001EO5QW8"
product_id = st.sidebar.text_input("Enter Product ID Below, Then Click \"Analyze Product\"", test_product_id)
#product_id = st.sidebar.text_input("Enter Product ID:")

# --- Main Content Area ---
if st.sidebar.button("Analyze Product"):
    if product_id:  # Check if product_id is not empty
        st.header(f"Analysis for Product: {product_id}")

        # Get product stats and write them to the screen
        product_stats = get_product_stats(product_id)
        if product_stats: #check if product_stats is not none
            st.json(product_stats)

            # Get topics and write them to the screen
            print(product_stats)
            if "key_themes" in product_stats and product_stats["key_themes"] is not None: #verify that key_themes exists and is not none.
                st.write(product_stats["key_themes"])
            else:
                st.write("Key themes not available for this product.")
        else:
            st.write("Product not found.")
    else:
        st.warning("Please enter a product ID.")

# --- Footer ---
st.markdown("---")
st.write("Developed by Josh Houlding")