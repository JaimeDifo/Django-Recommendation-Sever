import json
import pandas as pd
from django.http import HttpResponse
from response import Response
from rest_framework import status
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Load data from CSV file
data = pd.read_csv('/Users/jaimedifolorenzo/Desktop/Products.csv')

# Convert ingredient lists to string format
data['Ingredients'] = data['Ingredients'].apply(lambda x: ' '.join(eval(x)))

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words='english')

# Fit vectorizer on recipe ingredients
vectorizer.fit(data['Ingredients'])

# Transform recipe ingredients into feature vectors
features = vectorizer.transform(data['Ingredients'])


# Get indices of recipes similar to a given ingredient
def get_similar_recipes(request, ingredient):
    # Create a new DataFrame with only the recipes containing the given ingredient
    selected_recipes = data[data['Ingredients'].str.contains(ingredient.lower())]
    if len(selected_recipes) == 0:
        return "No recipes found containing " + ingredient
    # Transform selected recipe ingredients into feature vectors
    selected_features = vectorizer.transform(selected_recipes['Ingredients'])
    # Calculate cosine similarity between selected feature vectors and all feature vectors
    similarity_scores = cosine_similarity(selected_features, features)
    # Get indices of top 5 recipes with highest similarity scores
    indices = similarity_scores.argsort()[0][-5:]
    # Get top 5 recipe names
    recommended_recipes = data['Id'].iloc[indices].values.tolist()
    recipes_string = json.dumps(recommended_recipes[::-1])
    return HttpResponse(recipes_string)


from django.shortcuts import render
#python manage.py runserver 8080

