import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# Data file names
users_filename = "data/BX-Users.csv"
books_filename = "data/BX-Books.csv"
ratings_filename = "data/BX-Book-Ratings.csv"

pp = pprint.PrettyPrinter(indent=4).pprint

# Import CSV data into dataframes
df_users = pd.read_csv(
    users_filename,
    encoding="ISO-8859-1",
    sep=";",
    header=0,
    names=["user_id", "loc", "age"],
    usecols=["user_id", "loc", "age"],
    dtype={"user_id": "int32", "loc": "str", "age": "Int32"},
)

df_books = pd.read_csv(
    books_filename,
    encoding="ISO-8859-1",
    sep=";",
    header=0,
    names=["isbn", "title", "author"],
    usecols=["isbn", "title", "author"],
    dtype={"isbn": "str", "title": "str", "author": "str"},
)

df_ratings = pd.read_csv(
    ratings_filename,
    encoding="ISO-8859-1",
    sep=";",
    header=0,
    names=["user_id", "isbn", "rating"],
    usecols=["user_id", "isbn", "rating"],
    dtype={"user_id": "int32", "isbn": "str", "rating": "int32"},
)


def clean_dataframes(*dataframes):
    for df in dataframes:
        df.drop_duplicates(inplace=True)
        df.dropna(inplace=True)


clean_dataframes(df_users, df_books, df_ratings)

pd.set_option("display.max_columns", None)
user_counts = df_ratings["user_id"].value_counts()
isbn_counts = df_ratings["isbn"].value_counts()

frequent_users = user_counts[user_counts >= 200].index
popular_books = isbn_counts[isbn_counts >= 100].index

df_ratings = df_ratings[
    df_ratings["user_id"].isin(frequent_users) & df_ratings["isbn"].isin(popular_books)
]

# Merge with df_books[popular_books] subset to get book titles
df_ratings = pd.merge(
    df_ratings, df_books[df_books["isbn"].isin(popular_books)], on="isbn"
)

assert df_ratings["rating"].between(0, 10).all(), "Ratings outside valid range"
assert df_ratings["user_id"].isin(frequent_users).all(), "Invalid user_id"
assert df_ratings["isbn"].isin(popular_books).all(), "Invalid ISBN"

# print(df_ratings["rating"].describe())
# Index(['user_id', 'isbn', 'rating', 'title', 'author'], dtype='object')


# Function to return recommended books
def get_recommends(book=""):
    # 1. Create the ratings pivot table
    ratings_pivot = df_ratings.pivot(
        index="isbn", columns="user_id", values="rating"
    ).fillna(0)

    # 2. Fit KNN with the sparse matrix
    knn = NearestNeighbors(metric="cosine", n_neighbors=6, n_jobs=-1)
    knn.fit(csr_matrix(ratings_pivot))

    # 3. Get the book's ISBN
    book_isbn = df_books[df_books["title"] == book]["isbn"].values[0]

    # 4. Get the book's ratings vector (keeping the same structure as training data)
    book_vector = ratings_pivot.loc[[book_isbn]]  # Keep as DataFrame with same columns

    # 5. Find nearest neighbors
    distances, indices = knn.kneighbors(csr_matrix(book_vector))
    pp(distances)
    pp(df_books[df_books["isbn"].isin(ratings_pivot.index[indices.flatten()])]["title"])

    # 6. Get recommende books
    recommended_isbns = ratings_pivot.index[indices.flatten()]
    recommended_books = df_books[df_books["isbn"].isin(recommended_isbns)]
    # Return books and distances, sorted by distance

    recommended_books = [
        book,
        [
            [x, float(y)]
            for x, y in zip(
                df_books[df_books["isbn"].isin(recommended_isbns)]["title"],
                distances.flatten(),
            )
        ],
    ]
    recommended_books[1].sort(key=lambda x: x[1], reverse=False)
    # Convert to np to std float
    return recommended_books


# Test the recommendation function
books = get_recommends("Where the Heart Is (Oprah's Book Club (Paperback))")
pp(books)


def test_book_recommendation():
    test_pass = True
    recommends = get_recommends("Where the Heart Is (Oprah's Book Club (Paperback))")
    if recommends[0] != "Where the Heart Is (Oprah's Book Club (Paperback))":
        test_pass = False
    recommended_books = [
        "I'll Be Seeing You",
        "The Weight of Water",
        "The Surgeon",
        "I Know This Much Is True",
    ]
    recommended_books_dist = [0.8, 0.77, 0.77, 0.77]
    for i in range(2):
        if recommends[1][i][0] not in recommended_books:
            test_pass = False
        if abs(recommends[1][i][1] - recommended_books_dist[i]) >= 0.05:
            test_pass = False
    if test_pass:
        print("You passed the challenge! 🎉🎉🎉🎉🎉")
    else:
        print("You haven't passed yet. Keep trying!")


# test_book_recommendation()
