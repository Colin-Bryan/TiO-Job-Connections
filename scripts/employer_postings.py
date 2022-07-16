# Import libraries
import pandas as pd
from bs4 import BeautifulSoup
import requests

# Create employer postings dataframe and return URLs
def get_employer_postings():
    
    # Specify path of employer postings:
    # CB 7.16 - In future, read this from hosted db. For now, Excel file will suffice
    path = "data\TiO - Employer Partner Job Postings.xlsx"  

    # Read file path into dataframe
    postings_df = pd.read_excel(path)

    # Drop NaNs
    postings_df.dropna(subset = ['Opening URL'], inplace = True)
    
    # Save URLs into a list
    url_list = list(postings_df['Opening URL'])

    return postings_df, url_list

def scrape_indeed_postings(url_list):
    
    # Create list of indeed_postings onl
    indeed_url_list = []
    
    # Get list of posts from Indeed
    for url in url_list:
        if 'indeed.com' in url:
            indeed_url_list.append(url)
    
    # Create lists to store data
    job_titles = []
    job_locations = []
    job_descriptions = []

    # Get data from each Indeed URL:    
    for url in indeed_url_list:
        
        # Get html data from the URL
        html_data = requests.get(url).text
        
        # Pass into parser
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
    indeed_dict = {'Title':job_titles, 'Location':job_locations, 'Description':job_descriptions, 'URL':indeed_url_list}

    # Create dataframe
    indeed_df = pd.DataFrame(indeed_dict)
    
    # Drop rows that didn't return results
    indeed_df = indeed_df[indeed_df['Description'] != 'Could not find description'].reset_index(drop=True)
    
    return indeed_df 