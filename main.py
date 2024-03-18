from google.cloud import storage
from google.cloud import bigquery
import io

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def affinity_scoring(request):
    request_json = request.get_json(silent=True)
    request_args = request.args

    if request_json and 'file_name' in request_json:
        file_name = request_json['file_name']
    elif request_args and 'file_name' in request_args:
        file_name = request_args['file_name']
    else:
        return 'Error: File name not provided.'

    # Read CSV file
    df = read_csv_file(file_name)

    # Filter data for engagements in 2023
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'].dt.year == 2023].reset_index(drop=True) 
     
    # engagement_id
    df['engagement_id'] = df.index
    
    # Cleanup
    df['content_id'] = df['source_system'] + '|' + df['content_id']
    df = df.drop(columns=['country', 'source_system'])
    
    print('Before splitting topics:')
    print('records: ', len(df))
    print(df.nunique())
    
    # Splitting topics and creating new rows
    df = df.assign(topic=df['topic'].str.split(';')).explode('topic').reset_index(drop=True)
    
    print('After splitting topics:')
    print('records: ', len(df))
    print(df.nunique())
    
    # Scoring
    all_topics = calculate_topic_summary(df)
    
    # sort topics by number of users
    topics = all_topics['topic'] 
    
    #loop through topics and do the scoring
    all_user_scores = pd.DataFrame()
    all_gradings = pd.DataFrame()
    
    cnt = 0
    
    for topic in topics:
        #print(topic)
        
        user_scores = calculate_user_scores(df, topic)
        grading = calculate_grading(user_scores)
        
        plot_user_scores(user_scores, topic, str(cnt) + '_' + topic + '.jpg')
        cnt += 1
        
        all_user_scores = pd.concat([all_user_scores, user_scores], ignore_index=True)
        all_gradings = pd.concat([all_gradings, grading], ignore_index=True)
        
            
    print('User Scores - rows: ',  len(all_user_scores))
    print('Grading - rows: ', len(all_gradings))
    
    # Write DataFrames to a CSV files in the scoring_output bucket
    write_to_csv(all_user_scores, 'user_scores.csv')
    write_to_csv(all_gradings, 'grading.csv')
    write_to_csv(all_topics, 'topics.csv')
 
    # Create BigQuery tables
    df2bq(all_user_scores, 'user_scores')
    df2bq(all_gradings, 'grading')
    df2bq(all_topics, 'topics')
    
    print('Success')
    return 'Success'


def calculate_topic_summary(df):
    # Group by 'topic' and calculate aggregated metrics
    topic_summary = df.groupby('topic').agg(
        total_contents=('content_id', 'nunique'),
        engagements=('engagement_id', 'nunique'),
        total_users=('scv_id', 'nunique')
    ).reset_index()

    # Add a summary row for overall statistics
    overall_summary = pd.DataFrame({
        'topic': ['overall'],
        'total_contents': df['content_id'].nunique(),
        'engagements': df['engagement_id'].nunique(),
        'total_users': df['scv_id'].nunique()
    })

    # Concatenate overall summary row with topic summary DataFrame
    topic_summary = pd.concat([topic_summary, overall_summary], ignore_index=True).sort_values(by='total_users', ascending=False).reset_index(drop=True)

    return topic_summary


def calculate_user_scores(df, topic):
    # Filter dataframe based on topic
    if topic == 'overall':
        filtered_df = df
    else:
        filtered_df = df[df['topic'] == topic]

    # Group by 'scv_id' and count number of engagements
    engagements_per_user = filtered_df.groupby('scv_id')['engagement_id'].nunique()
    
    # Count the number of users for each count of engagements
    user_counts = engagements_per_user.value_counts().reset_index()
    user_counts.columns = ['Engagement Count', 'Number of Users']

    # Define the number of score categories (bins)
    num_bars = 10

    # Create bins for grouping lower count engagements
    bins = [0] + [user_counts['Engagement Count'].quantile(q) for q in [(i+1)/num_bars for i in range(num_bars-1)]] + [user_counts['Engagement Count'].max()]

    if bins == [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]:
        bins = [0, 1]
        #print(bins)
    
    # Assign score to each user
    user_scores = pd.DataFrame({'scv_id': engagements_per_user.index,
                                'engagements': engagements_per_user.values,
                                'score': pd.cut(engagements_per_user, bins, labels=False)})
    user_scores['score'] += 1
    
    if bins == [0, 1]:
        user_scores['score'] = user_scores['score'] * 10

    # Add topic column
    user_scores['topic'] = topic

    # Include users excluded by filtering with score 0 and 0 engagements
    if topic != 'overall':
        excluded_users = df[~df['scv_id'].isin(filtered_df['scv_id'])]['scv_id'].unique()
        excluded_df = pd.DataFrame({'scv_id': excluded_users, 'engagements': 0, 'score': 0, 'topic': topic})
        user_scores = pd.concat([user_scores, excluded_df], ignore_index=True)
    
    return user_scores

def calculate_grading(user_scores):
    # Group by score and find the range of engagements for each score category
    grading = user_scores.groupby(['topic', 'score'])['engagements'].agg(['min', 'max', 'count', 'sum']).reset_index()
    grading.columns = ['topic', 'score', 'engagements_from', 'engagements_to', 'total_users', 'total_engagements']
    grading['score'] = grading['score'].astype(int)
    
    # Adjust engagements_from for the score 1 limit
    grading.loc[grading['score'] == 1, 'engagements_from'] = 1  # Adjust for 1-based scoring

    # Adjust engagements_from for other scores
    #for i in range(1, len(grading)):
    #    grading.loc[i, 'engagements_from'] = grading.loc[i - 1, 'engagements_to'] + 1

    return grading

def plot_user_scores(user_scores, topic, filename):
    # Restart plt
    plt.close()
    
    # Plot
    aggregated = user_scores.groupby('score')['scv_id'].nunique()
    ax = aggregated.plot(kind='bar')
    plt.title(f'Number of Users by Score for Topic: {topic}')
    plt.xlabel('Score')
    plt.ylabel('Number of Users')
    plt.xticks(rotation=0)
    
    # Annotate each bar with its value
    for i, value in enumerate(aggregated):
        ax.text(i, value, str(value), ha='center', va='bottom')

    # Save image to bucket
    save_plot_to_gcs(plt, filename)
    


#GOOGLE CLOUD SPECIFIC FUNCTIONALITIES

def read_csv_file(file_name):
    # Initialize Google Cloud Storage client
    storage_client = storage.Client()

    # Retrieve the bucket
    bucket = storage_client.bucket('user_engagement_input')

    # Download the file content as bytes, decode it, and read into DataFrame
    blob = bucket.blob(file_name)
    csv_content = blob.download_as_string().decode('utf-8')
    df = pd.read_csv(io.StringIO(csv_content))

    return df

def write_to_csv(df, output_file):
    # df to csv
    csv_content = df.to_csv(index=False)

    # Initialize Google Cloud Storage client
    storage_client = storage.Client()

    # Retrieve the bucket
    bucket = storage_client.bucket('scoring_output')

    # Upload the CSV file to the bucket
    blob = bucket.blob(output_file)
    blob.upload_from_string(csv_content)

def df2bq(df, table_name):
    # Initialize BigQuery client
    bq_client = bigquery.Client()

    # Define BigQuery dataset and table references
    dataset_ref = bq_client.dataset('scoring')
    table_ref = dataset_ref.table(table_name)

    # Define job configuration
    job_config = bigquery.LoadJobConfig(
        write_disposition='WRITE_TRUNCATE'
    )

    # Load DataFrame into BigQuery table
    job = bq_client.load_table_from_dataframe(df, table_ref, job_config=job_config)
    job.result()  # Wait for the job to complete

    print(f'Loaded {len(df)} rows into BigQuery table {table_name}')
    
    
def save_plot_to_gcs(plt, file_name):

    # Save the plot to a BytesIO buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='jpg')
    buffer.seek(0)
    
    # Initialize Google Cloud Storage client
    storage_client = storage.Client()
    
    # Retrieve the bucket
    bucket = storage_client.bucket('scoring_output')
    
    # Create a blob representing the path to save the plot in the bucket
    blob = bucket.blob(f'graphs/{file_name}')
    
    # Upload the BytesIO buffer to the bucket
    blob.upload_from_file(buffer, content_type='image/jpg')