def multilabel_preproc(reviews, restaurants):
    '''
    Preprocessing for multilabel classification
    '''
    # Drops unnecessary columns
    reviews.drop(['Reviewer', 'Metadata', 'Time', 'Pictures', 'Rating'], axis=1, inplace=True)

    restaurants.drop(['Links', 'Cost', 'Collections', 'Timings'], axis=1, inplace=True)

    # Merges the cuisines column with the reviews
    reviews = reviews.merge(restaurants[['Name', 'Cuisines']], 
                                        left_on='Restaurant', right_on='Name', 
                                        how='left').drop(columns=['Name'])

    # Cleanup
    reviews.drop(['Restaurant'], axis=1, inplace=True)
    reviews.dropna(subset=['Review'], inplace=True)

    # Splits the Cuisines by commas
    reviews['Cuisines'] = reviews['Cuisines'].apply(lambda x: [cuisine.strip() for cuisine in x.split(',')])

    return reviews

def sentiment_preproc(reviews):
    '''
    Prepares the raw reviews dataset for the sentiment analysis task
    Since the requirement is to use review polarity to predict Zomato score, the remaining columns are no longer needed for this task
    Note that we assume the Zomato score as the average of all reviews of this restaurant.
    '''
    # Drops unnecessary columns and null rows
    reviews.drop(['Reviewer', 'Metadata', 'Time', 'Pictures'], axis=1, inplace=True)
    reviews.dropna(subset=['Review', 'Rating'], inplace=True)
  
    # Converts the rating column to numeric
    reviews['Rating'] = pd.to_numeric(reviews['Rating'], errors='coerce')

    # Groups by restaurant and calculates the mean rating
    average_ratings = reviews.groupby('Restaurant')['Rating'].mean().reset_index()
    average_ratings.rename(columns={'Rating': 'Zomato Score'}, inplace=True)

    # Merges this with the original df to add the column
    reviews = reviews.merge(average_ratings, on='Restaurant', how='left')

    # Removes the unneeded restaurant column
    reviews.drop(['Restaurant', 'Rating'], axis=1, inplace=True)

    return reviews