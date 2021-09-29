# movie-recommender

Using python and its libraries i.e pandas, numpy, Scikit-learn, a movie recommender system has been made.
According to attributes of user input movie ( movie title), top 10 movies similar to that will be displayed.
The entire dataset is filtered at each step on the basis of values of categorical and numerical columns of user input title.
After filtering data to a certain extent based on related attributes, TF-IDF technique and cosine_similarity on genre and description features is used to get the final result.
