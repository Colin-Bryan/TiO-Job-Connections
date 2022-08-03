# Import libraries
import pandas as pd
from bs4 import BeautifulSoup
import requests
from datetime import datetime
import os

# Import functions from other scripts
from scripts.process_text import AnalyzeText

### GCP Imports and Setup ###
import google.auth
from google.cloud import storage
from io import BytesIO

# Create employer postings dataframe and return URLs
def get_employer_postings(gcp_storage_bucket):
    '''
    Reads the employer partner job postings data in Excel and returns a DataFrame of relevant postings to parse.

        Parameters:
            gcp_storage_bucket (Object): The bucket where GCP data is stored for processing 
        
        Returns:
            postings_df (DataFrame): A DataFrame of relevant job postings to parse.
    '''
    
    ## Load Employer Partner Job Postings data with GCP
    try:
        # Get blob
        blob = storage.blob.Blob('TiO - Employer Partner Job Postings.xlsx', gcp_storage_bucket)

        # Get content
        content = blob.download_as_string()

        # Read into dataframe
        postings_df = pd.read_excel(BytesIO(content))
    
    except:
        st.error('No jobs found in database. Please update postings')

    # Drop NaNs in URL
    postings_df.dropna(subset = ['Opening URL'], inplace = True)

    # Drop duplicate URLs
    postings_df.drop_duplicates(subset = ['Opening URL'], inplace = True)
    
    return postings_df

def tokenize_postings(df, gcp_storage_bucket):
    '''
    Alters the DataFrame of relevant job postings and adds a column with tokenized text.

    Parameters:
        df (DataFrame): A dataframe of job postings to read and tokeneize specific columns.
        gcp_storage_bucket (Object): The bucket where GCP data is stored for processing 
    
    Returns:
        df (DataFrame): The original dataframe with an additional column "processed_text" that contains
                        tokenized text.
    '''
    # Load in AnalyzeText class
    anlyz_txt = AnalyzeText()

    # Create processed_text column to store tokenized text by using tokenize_text function
    df['processed_text'] = df['full_text'].apply(lambda x: anlyz_txt.tokenize_text(x, gcp_storage_bucket))

    # # Output job postings to CSV for archiving
    blob = storage.blob.Blob('postings/archive/archived_postings/Job Postings_{}.csv'.format(datetime.now().strftime("%Y-%m-%d")), gcp_storage_bucket)
    blob.upload_from_string(df.to_csv(index=False), 'text/csv')

    # Create most recent job postings for processing
    blob = storage.blob.Blob('postings/Job Postings.csv', gcp_storage_bucket)
    blob.upload_from_string(df.to_csv(index=False), 'text/csv')

    # Return processed dataframe
    return df

def process_URL_postings(postings_df, gcp_storage_bucket):
    '''
    Defines the criteria for job postings to scrape and gathers relevant data for downstream modeling.

    Parameters:
        postings_df (DataFrame): A DataFrame of job postings to process.
        gcp_storage_bucket (Object): The bucket where GCP data is stored for processing 
    
    Returns:
        scraped_df (DataFrame): A DataFrame of the job postings with an additional tokenized column of text for modeling.
    '''

    # Define filter criteria for postings_df
    # CB 7.24 - Indeed only for v1
    url_filter_criteria = ['indeed.com']

    # Create sub dataframe to process jobs
    sub_postings_df = postings_df[postings_df['Opening URL'].str.contains('|'.join(url_filter_criteria))]
    
    # Create lists to store data while looping
    job_titles = []
    job_locations = []
    job_descriptions = []

    # Get data from each Indeed URL:    
    for url in list(sub_postings_df['Opening URL']):
        
        # Get html data from the URL
        html_data = requests.get(url).text
        
        # Pass into BeautifulSoup parser
        soup = BeautifulSoup(html_data, 'html.parser')
        
        # Save job titles and locations into list
        try:
            # Get page title
            page_title = soup.title.get_text(strip = True)
            
            # Split page title to get job title and location
            job_title = page_title.split(' - ')[0]            
            location = page_title.split(' - ')[1]
            
            # If there is a hyphen in the job title, resplit. 
            # Hypothesis is that a location will be missing a comma if there is a hypen in the title and it's not remote
            if ',' not in location:
                if 'remote' not in location.lower():
                    # Split page title to get job title and location
                    job_title = page_title.split(' - ')[0] + ' - ' + page_title.split(' - ')[1]         
                    location = page_title.split(' - ')[2]
            
            # Append title and location to lists
            job_titles.append(job_title)
            job_locations.append(location)
            
        except:
            # Put catchall string in fields if error
            job_titles.append('Could not find title')
            job_locations.append('Could not find location')
        
        # Save job descriptions into list
        try:
            job_descriptions.append(
                soup.select_one("#jobDescriptionText").get_text(strip=True, separator="\n")
            )
        except:
            job_descriptions.append('Could not find description')
            
        
    # Package up everything into dataframe by creating dictionary first
    scraped_dict = {'Employer':list(sub_postings_df.Employer),'Title':job_titles, 'Location':job_locations, 
            'Description':job_descriptions, 'URL':list(sub_postings_df['Opening URL'])}
   
    # Create dataframe
    scraped_df = pd.DataFrame(scraped_dict)
    
    # Drop rows that didn't return results
    scraped_df = scraped_df[scraped_df['Description'] != 'Could not find description'].reset_index(drop=True)

    # Add "Source" column - defaulting Indeed
    scraped_df['Source'] = 'Indeed'

    # For troubleshooting
    print(scraped_df.shape)


    # Combine job title, job location, and job description into full_text column to use as input for model
    scraped_df['full_text'] = scraped_df.apply(lambda x: ' '.join([x['Title'],x['Location'],x['Description']]),axis=1)

    # Tokenize postings and return processed dataframe
    return tokenize_postings(scraped_df, gcp_storage_bucket)

