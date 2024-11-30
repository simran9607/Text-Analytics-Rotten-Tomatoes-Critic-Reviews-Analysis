# Sentiment Analysis for Rotten Tomato Critic Reviews
INTRODUCTION
1. Business Problem
The key business problem involves leveraging movie and critic data from Rotten Tomatoes to gain
insights into the relationship between audience ratings (audience score) and critic reviews
(Tomatometer). The goal is to understand how these ratings impact a movie's performance, public
reception, and overall success.
Businesses, such as movie production studios, streaming platforms, and marketing agencies, can use
these insights for content strategy, marketing focus, and better targeting for releases.
2. Business Understanding
In the competitive world of film and entertainment, understanding the dynamics between audience
perceptions and critical evaluations is crucial for success. The business problem at hand involves
harnessing data from Rotten Tomatoes to explore the interplay between audience ratings and critic
reviews. By examining the relationship between the audience score and the Tomatometer rating, this
analysis aims to shed light on how these metrics influence a movie's performance, public reception,
and overall success.
The data used for this analysis is sourced from Rotten Tomatoes, a premier platform for movie ratings
and reviews. The dataset comprises two main components: the Movies Dataset and the Critics
Dataset. The Movies Dataset includes comprehensive details about each film, such as the movie title,
description, genres, duration, director, actors, and both audience and critic ratings. On the other hand,
the Critics Dataset captures detailed information about individual critic reviews, including the critic's
name, the publication where the review appeared, the review's publication date, the critic's score, and
the content of the review.
This data is essential for businesses such as movie production studios, streaming platforms, and
marketing agencies. By analysing these insights, these stakeholders can refine their content strategies,
optimize marketing efforts, and better target their release schedules to align with audience preferences
and critical feedback.
3. Data Description:
rotten_tomatoes_link: Link to the critic’s review of the movie, for source validation.
Critic Name: The name of the critic who reviewed the movie.
Review Publication: Where the review was published (e.g., newspapers, magazines, blogs).
Date: The date the review was published, useful for identifying trends over time.
Review Score: The critic’s score for the movie, which can be compared with the tomatometer
score and users’ ratings.
Review Content: The actual content of the review, which can be analyzed for sentiment or key
themes.
Dataset Link: https://www.kaggle.com/datasets/stefanoleone992/rotten-tomatoes-movies-and-critic-
reviews-dataset?select=rotten_tomatoes_critic_reviews.csv
2
Data Cleaning and Preprocessing:
1. Dropping Unnecessary Columns:
o Dropping columns ‘Length’ and ‘Actual sentiment’.
2. Text Preprocessing:
o The raw text data is cleaned and normalized by removing special characters, links, and
stop words (common words like "the," "is"). The text is also converted to lowercase.
o Lemmatization is applied to reduce words to their base forms (e.g., "running" becomes
"run").
3. Frequency Analysis:
o After preprocessing, the frequency of each unique word in the dataset is calculated.
o This frequency information is stored in a DataFrame and sorted to identify the most
frequently occurring terms.
4. Data Visualization: The results provide insights into the most common words used in the
text data, allowing for further analysis of the language patterns and themes present in the
dataset.
Positive Sentiment Text Distribution-
3
Negative Sentiment Text Distribution-
Neutral Sentiment Text Distribution
4
Overall Sentiment Text Distribution
5
EXPLORATORY DATA ANALYSIS
Data Visualization of Review Score
 The bar plot offers a clear view of how reviews are distributed across various rating scores.
The majority of reviews fall within the 4.0 rating range.
 The 2.0 rating is the second most common, followed by 3.0.
 Ratings of 0.0, 1.0, and 5.0 are relatively rare, indicating that extreme scores, whether high or
low, are uncommon in the dataset.
The table displays a subset of the dataset after several preprocessing steps, such as removing NaN
values and standardizing the review scores.
 The review scores have been converted to a consistent scale (e.g., 3.0, 4.0), making it easier
to compare reviews across different ratings.
 This transformation helps normalize the data for more accurate analysis.
 The "Length" column likely represents the length of the "review content," which is used to
visualize the distribution of review lengths.
 The Data Frame includes specific examples of critic names, their corresponding review
scores, and the actual review content.
6
Review length distribution for Different Ratings:
The plot consists of a grid of histograms illustrating how review lengths vary across different rating
scores (ranging from 0.0 to 5.0). This visualization offers insights into the relationship between
review length and rating behaviour. Here's a detailed breakdown:
Histograms:
Each histogram shows the distribution of review lengths (x-axis) and the count of reviews (y-axis) for
each rating. The bars represent the number of reviews for specific length ranges.
KDE Curves:
 The smooth lines overlaying the histograms are Kernel Density Estimate (KDE) curves,
which estimate the probability density of review lengths.
Key Observations:
 Rating 0.0, 1.0, and 5.0: These ratings have relatively few reviews, as seen from the low
counts on the yaxis. The distributions appear flatter due to the smaller number of reviews.
 Ratings 2.0, 3.0, and 4.0: These ratings have more reviews, with a higher count and more
well-defined distributions.
 Rating 4.0: Reviews for this rating tend to be somewhat longer, with a peak in the 2030 length
range.
 Rating 3.0: This rating has a broader range of review lengths, with a peak around 20
characters/words
 . Rating 5.0: Most reviews for this rating seem to be short, with very few reviews over 30 in
length.
7
SENTIMENT ANALYSIS
1. Sentiment Analysis using VADER
We will perform sentiment analysis to analyse sentiment of each review and classify it as positive,
negative, or neutral. And results will be compared with the real rating of the restaurants.
The image is a confusion matrix visualizing the performance of a sentiment analysis model.
Rows (Actual Sentiment): These correspond to the actual sentiment of the data points (Negative,
Neutral, Positive).
Columns (Predicted Sentiment): These represent the predicted sentiment categories from the model.
Diagonal Cells (True Positives): Values on the diagonal (506 for Negative, 145 for Neutral, 797 for
Positive) represent the correct predictions made by the model. These are where the predicted
sentiment matches the actual sentiment.
8
Off Diagonal Cells (False Positives and False Negatives):
For example, 270 in the top center means that 270 Negative reviews were incorrectly predicted as
Neutral.
498 in the top right means 498 Negative reviews were wrongly predicted as Positive.
Colour Scale: The blue color intensifies as the numbers increase, with darker shades representing
higher values. For instance, the darkest blue is in the bottomright cell, where 797 Positive reviews
were correctly predicted.
The confusion matrix for sentiment analysis shows how accurately the model predicted sentiment
categories (Negative, Neutral, Positive). Diagonal values indicate correct predictions, while off
diagonal values represent misclassifications. The model performs best in predicting Positive
sentiments, with 797 correct predictions, but has difficulty with Neutral sentiments, showing a high
number of misclassifications.
2. Sentiment Analysis using NRClex (NRC Emotion Lexicon)
A sentiment and emotion analysis of movie reviews using NRClex was conducted, showing predicted
and actual sentiments along with the proportions of emotions like fear, anger, trust, anticipation, joy,
and sadness for each review. Some reviews lacked certain emotions, resulting in `NaN` values. This
analysis helps determine the emotional tone and whether predicted sentiments align with actual ones.
This results in a data frame that includes both the original review data and the corresponding
sentiment and emotion frequencies for each review after analyzing sentiments and emotions using
NRClex.
9
DATA MODELING
1. Transforming the Dataset Using TF-IDF Vectorizer
2. Splitting the Data
3. Building the Model
Logistic Regression-
 The model performs well in predicting positive sentiments but struggles with neutral and
negative sentiments.
 The low recall for neutral and negative indicates that the model misses many instances of
these classes.
2. Linear Support Vector Classification-
 The model does a reasonably good job in predicting positive sentiments but struggles with
negative and especially neutral sentiments.
 The recall for neutral (0.23) is particularly low, indicating that the model has difficulty
identifying most neutral cases.
10
3. Random Forest-
 The model performs well in predicting positive sentiments, but struggles significantly with
negative and neutral sentiments.
 The recall for negative (0.19) and neutral (0.09) is particularly low, indicating that the model
misses most of the instances in these classes, even though it has decent precision.
 The low F1-scores for negative (0.31) and neutral (0.16) indicate the model's poor overall
performance in detecting these sentiments.
4. Naive Bayes Multinominal
 The model performs very well in predicting positive sentiments but performs poorly with
negative and neutral sentiments.
 The recall for negative (0.05) and neutral (0.04) is extremely low, meaning the model rarely
identifies these classes, despite having good precision.
COMPARING MODELS PERFORMANCE
SVC is the best-performing model in this case with an accuracy of about 60%.
11
Topic Modelling
The objective is to identify hidden themes or topics that the text data naturally clusters into.
 Preprocessing: The raw text data is being cleaned by removing numbers, extra spaces, and
converting the text to lowercase. Stopwords are also removed. This results in a cleaner set of
words for analysis.
 Vectorization: The cleaned text is transformed into numerical form using a technique called
Count Vectorization. This creates a matrix where each row represents a document (review), and
each column represents a word (or word pair, as we are considering n-grams). The matrix
captures the frequency of each word or word pair appearing in the reviews.
 LDA Model (Latent Dirichlet Allocation): LDA identifies a pre-defined number of topics (in
this case, 5) from the vectorized data. It attempts to group the words in the dataset into clusters or
topics, where each topic is represented by a collection of words that frequently occur together.
 Identifying Topics: After fitting the LDA model, the algorithm generates a list of words that are
most representative of each topic. These words help to interpret and label the topics, based on the
patterns that emerge from the data.
 Visualization (Word Clouds): The size of the word in the cloud indicates its relevance to the
topic, with larger words being more important.
12
Text clustering for reviews
We can use text clustering algorithms, such as K-means, to group similar reviews into clusters based
on the similarity of their contents. The goal of clustering is to identify underlying patterns or
structures in the data, which can be useful for recommendation systems.
13
1. 2. 3. 4. 5. 6. 7. Selecting Data: We focus on reviews that have been given a rating of 2 or lower, as these are
considered "negative reviews".
Vectorization (TF-IDF): The textual data (review content) is converted into numerical form
using TF-IDF Vectorization. This technique captures the importance of words in each
review relative to the entire collection, giving more weight to words that are unique to
specific reviews.
Clustering (K-Means): The reviews are grouped into clusters using the K-Means algorithm,
where we pre-define the number of clusters (in this case, 3). K-Means tries to group reviews
into clusters where similar reviews are closer to each other based on their content.
Evaluating Cluster Performance: The quality of the clusters is evaluated using the
Adjusted Rand Index (ARI), which measures how well the predicted clusters (from K-
Means) align with the true labels (in this case, the review scores). A higher ARI score
indicates better clustering performance.
Dimensionality Reduction (PCA): To visualize the clustering, we reduce the high-
dimensional data into 2D using Principal Component Analysis (PCA). This allows us to
create a scatter plot to display the clusters in two dimensions for easier interpretation.
Top Features for Each Cluster: For each cluster, we identify the top terms or keywords that
define that cluster. These terms are selected based on their importance within the cluster.
Visualizing Clusters: A scatter plot is created to visually represent the clusters. Each review
is plotted, and different clusters are coloured differently, allowing us to see how well the
reviews are grouped together.
14
8. Word Clouds for Clusters:
9. Finding Optimal Number of Clusters: To determine the best number of clusters, we use the
Silhouette Score, which measures how well-separated the clusters are. The score is
calculated for different numbers of clusters, and a scree plot is generated to visualize which
number of clusters is optimal.
Through silhouette score we identify that optimal number of clusters are 3.
15
Business Strategy
1. Targeted Marketing Campaigns
 Leverage Positive Sentiment: Use the insights from positive sentiment analysis to craft
marketing campaigns that highlight the strengths of a movie. For films with high positive
sentiment among critics, emphasize these reviews in promotional materials to attract
audiences who value critical acclaim.
 Addressing Critic Concerns: For films that receive mixed or negative reviews, consider
addressing specific criticisms in marketing strategies or even during the film's promotional
tours. Acknowledge and engage with critical feedback to show responsiveness and
willingness to improve.
2. Strategic Release Timing
 Aligning with Audience Preferences: Use data from audience ratings and critic reviews to
strategically time the release of films. For example, movies with broad audience appeal but
mixed critical reviews might perform better during holiday seasons when audiences are more
diverse and less influenced by critics.
 Festival Circuits: Films with strong critic reviews should be prioritized for release during film
festivals or award seasons, where they are likely to gain more attention from both the public
and industry professionals.
3. Critic Engagement Strategies
 Build Relationships with Influential Critics: Identifying critics who consistently align with
audience preferences can be beneficial. Establishing relationships with these critics through
exclusive screenings or interviews may result in more favorable reviews and word-of-mouth
promotion.
 Utilize Feedback Loops: Create feedback loops where critic reviews are used to make
iterative improvements in film projects during the production phase. This can help in refining
the final product to better meet both critic and audience expectations.
4. Enhanced Audience Interaction
 Encourage User Reviews and Ratings: While critic reviews are important, audience scores
also play a critical role in a movie's success. Encourage viewers to leave reviews and ratings
on platforms like Rotten Tomatoes, as high audience engagement often correlates with better
overall performance.
6. Post-Release Analytics
 Monitor and Adjust: Continuously monitor critic reviews and audience ratings after the
release. Use this data to adjust ongoing marketing strategies, such as deciding which regions
to focus on or which demographic groups to target more aggressively.
 Long-Tail Strategy: For movies with delayed positive reception, a long-tail marketing
strategy can be effective. Continue promoting these films based on ongoing positive reviews,
potentially extending their run in theatres or on streaming platforms.
It can help movie production studios, streaming platforms, and marketing agencies better navigate the
complex relationship between critic reviews and audience reception.
16
CONCLUSION
The analysis conducted on Rotten Tomatoes critic reviews has provided valuable insights into the
relationship between audience ratings and critic reviews. By examining these dynamics, businesses in
the entertainment industry can refine their strategies to better align with public perception and critical
feedback. The sentiment analysis using VADER and NRClex revealed that while models perform
well in predicting positive sentiments, they struggle with neutral and negative sentiments. Among the
models tested, the Support Vector Classifier (SVC) showed the best performance with an accuracy of
approximately 60%.
Furthermore, the topic modeling and text clustering efforts helped uncover hidden themes and
patterns in the reviews, which could be leveraged for more targeted content strategies and marketing
efforts. The identification of three optimal clusters through the silhouette score further emphasizes the
diversity of opinions among critics, which can be used to tailor different aspects of a film's release and
marketing campaign.
Overall, this project highlights the importance of understanding the nuances of critic reviews and
audience scores, and how these insights can be harnessed for business success in the competitive film
industry.
