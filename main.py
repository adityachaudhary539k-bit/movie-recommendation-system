import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
movies=pd.read_csv("movies.csv")
tfidf=TfidfVectorizer(stop_words="english")
tfidf_matrix=tfidf.fit_transform(movies["Genre"].fillna(''))
cosine_sim=cosine_similarity(tfidf_matrix,tfidf_matrix)
movie_name= input("Enter movie name: ").lower().strip()
if not movies["Title"].str.lower().eq(movie_name).any():
    print("Movie not found! Please check your spelling.")
else:
    idx = movies[movies["Title"].str.lower()==movie_name].index[0]
    sim_scores=list(enumerate(cosine_sim[idx]))
    sim_scores=sorted(sim_scores,key=lambda x:x[1], reverse=True)
    top_movies= sim_scores[1:6]
    print("\n--- Top Recommendations ---")
    for index, score in top_movies:
        print(f"{movies['Title'].iloc[index]} (similarity:{score:.2f})")
