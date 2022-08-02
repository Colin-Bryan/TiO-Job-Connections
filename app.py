# Import libraries
import streamlit as st
import plotly.express as px
import pandas as pd
import os
import time

# Import functions and classes
from scripts.process_text import ExtractResumeText, AnalyzeText
from scripts.employer_postings import get_employer_postings, process_URL_postings

### GCP Imports and Setup ###
import google.auth
from google.cloud import storage
from io import BytesIO

# # Get service key for accessing Google Cloud (local only)
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'ServiceKey_GoogleCloud.json'

# Initialize storage client from GCP
storage_client = storage.Client()

# Access gcp bucket
gcp_storage_bucket = storage_client.get_bucket('tio-job-connections.appspot.com')


def main():
    '''
    Main function to run the Streamlit user interface.

    Users have the option of uploading a resume to compare to job postings, 
    viewing existing job postings, or updating the job postings database to get
    new features for comparison to resumes.
    '''

    # Create page config
    st.set_page_config(page_title="Job Connections", page_icon =":handshake:")#,layout="wide")

    # Streamlit title 
    st.title(":handshake: Find Job Opportunities")

    # Initilalize resume form holder as empty element
    resume_form_holder = st.empty()

    # Initialize Empty Element to hold Screen Content
    screen_content_holder = st.empty()

    # Initialize form to hold file uploader and submit button
    with resume_form_holder.form('file-upload', clear_on_submit=True):
        st.subheader('📊 Upload a resume to compare to active job postings')
        uploaded_file = st.file_uploader("Microsoft Word (.DOCX) format preferred", type=["docx","PDF"], accept_multiple_files = False)
        submitted = st.form_submit_button("Process")

    # Initialize AnalyzeText class
    anlyz_txt = AnalyzeText()

    #### Begin Functionality. Throw error message if form is submitted without a file being uploaded
    if submitted and uploaded_file is None:
        screen_content_holder.error('Please upload a resume')

    # If file is uploaded and form is submitted, process resume
    elif submitted and uploaded_file is not None:

        # Hide resume form
        resume_form_holder.empty()

        print('Resume processing start')

        # First, Scrape Resume and Extract Text
        with screen_content_holder.container():

            # First, Scrape Resume and Extract Text
            with st.spinner("Processing resume to find matches..."):
                # Load in job postings existing in data folder to a dataframe

                try:
                    ## Load data with GCP
                    # Get blob
                    blob = storage.blob.Blob('postings/Job Postings.xlsx', gcp_storage_bucket)

                    # Get content
                    content = blob.download_as_string()

                    # Read into dataframe
                    jobs_df = pd.read_excel(BytesIO(content))
                
                except:
                    st.error('No jobs found in database. Please update postings')

                ##### Initialize ExtractResumeText class #####
                ext_txt = ExtractResumeText()

                # Extract text from resume
                raw_text = ext_txt.get_resume_text(uploaded_file)

                # Get person's name
                name = ext_txt.get_name(raw_text)

                # Get person's email
                email = ext_txt.get_email(raw_text)

                # Get person's phone number
                phone = ext_txt.get_phone(raw_text)

                # Get person's location
                resume_location = ext_txt.get_location(raw_text, name)

                # CB 7.16 - Commenting out due to API limit
                # Get person's skills
                #skills = ext_txt.get_skills(raw_text)
                skills = ''

                # Append all extracted attributes to a dictionary
                # CB 7.24 - Placeholder for v2 usage
                attribute_dict = {'Name':name,'Email':email,'Phone':phone,
                                'Location':resume_location,'Skills':skills}

                ### Build Word Count Features ###
                # Create processed_text by tokening the raw resume text
                processed_text = anlyz_txt.tokenize_text(raw_text, gcp_storage_bucket)

                ### Analyze word count features and get similarity score from processed_text (Tf-idf and Bag-of-words)
                # Since it is a resume, it returns the resume features
                word_sim_results_df = anlyz_txt.build_and_analyze_word_count_features(gcp_storage_bucket, data = [processed_text], jobs_df = jobs_df, data_type='resume')

                ### Build and Analyze Resume Features with Sentence Transformer ###
                sent_trans_sim_df = anlyz_txt.analyze_with_transformer(gcp_storage_bucket, resume_data = raw_text, jobs_df = jobs_df, data_type='resume', name = name)

                # Display success message
                st.success(':smile: Matches found in employer database. Output file was created successfully.')
                st.subheader('Top 10 Matches for {}:'.format(name))

                # Join word count and transformer dataframes
                display_df = word_sim_results_df.merge(sent_trans_sim_df, left_on = ['Title'], right_on = ['Title'])

                # Calculate average score of Tf-idf, BoW, and Sentence Transformer similarities
                display_df['Average Score'] = display_df[['Tf-idf Score', 'BoW Score', 'Transformer Score']].mean(axis = 1)

                # Display dataframe with top 10 matches and select columns
                st.dataframe(display_df.loc[:9,
                                            ['Employer','Title','Tf-idf Score','BoW Score','Transformer Score','Average Score']
                                            ].sort_values(by = ['Average Score'], ascending = False).reset_index(drop=True))

                # Output to JSON format for TiO processing
                anlyz_txt.output_to_JSON(name, display_df, gcp_storage_bucket)
               
                print('Resume processing finished')
                
    ######## ---- INITIALIZE SIDEBAR ---- ########
    # Sidebar Image
    try:
        st.sidebar.image('https://storage.googleapis.com/tio-job-connections-static-images/TiO%20Logo.png')
    except:
        st.title('This is Opportunity')

    # Create Sidebar Title
    st.sidebar.title("Admin Settings")            

    # Create Upload Resume Button
    if st.sidebar.button('Upload New Resume'):
        # Hide screen content until resume is processed
        screen_content_holder.empty()

    # Create View Job Postings Button
    if st.sidebar.button('View Job Postings'):
        # Hide resume upload form
        resume_form_holder.empty()
        
        # Load in job postings existing in GCP to a dataframe
        try:
            # Get blob
            blob = storage.blob.Blob('postings/Job Postings.xlsx', gcp_storage_bucket)

            # Get content
            content = blob.download_as_string()

            # Read into dataframe
            jobs_df = pd.read_excel(BytesIO(content))

            # Replace holder element with container to show current postings
            with screen_content_holder.container():
                # Display count of current postings in database count
                st.subheader('Current Job Postings in Database: {}'.format(len(jobs_df)))

                # Display dataframe of postings
                st.dataframe(jobs_df.loc[:,['Employer','Title']].sort_values(by = ['Employer'], ascending = True))
                
        except:
            # Throw error
            st.error('No jobs found in database. Please update postings')
        

    # Create Update Job Postings Button
    if st.sidebar.button('Update Job Postings'):

        # Hide resume upload form
        resume_form_holder.empty()

        # Return list of posts as a dataframe and list of URLs to go scrape the web
        with screen_content_holder.container():

            with st.spinner("Updating Job Postings..."):

                # Process employer postings
                postings_df = get_employer_postings(gcp_storage_bucket)

                print('Job posting execution started')
                print('After retrieving initial job postings, the shape of the DataFrame is {}'.format(postings_df.shape))
 
            with st.spinner("Scraping job data from URLs..."):

                # CB 7.16 - Only scraping indeed postings as part of this project. Update to be more comprehensive in future
                # Processes and tokenizes URL postings
                jobs_df = process_URL_postings(postings_df, gcp_storage_bucket)

                print('After processing URLs, the shape of the DataFrame is {}'.format(jobs_df.shape))

            with st.spinner("Building Word Count Features..."):
                ### Build word count features from processed_text (Tf-idf and Bag-of-words) ###
                anlyz_txt.build_and_analyze_word_count_features(gcp_storage_bucket, data = jobs_df, data_type='jobs')

            with st.spinner("Creating Semantic Embeddings..."):
                ### Sentence Transformer for jobs ###
                anlyz_txt.analyze_with_transformer(gcp_storage_bucket, resume_data = "", jobs_df = jobs_df, data_type='jobs')
                
                print('Job posting execution finished')

                # Display success message
                st.success("✔️ Job data has been updated successfully")

                # When processing is finished, show job postings with tokenized text
                st.subheader('Current Job Postings in Database: {}'.format(len(jobs_df)))

                # Display dataframe
                st.dataframe(jobs_df.loc[:,['Employer','Title']].sort_values(by = ['Employer'], ascending = True))

# Execute main function
if __name__ == "__main__":
    main()
    # try:
    #     main()
    # except:
    #     st.error('An error occured. Please refresh the page and try again. If the problem persists, please contact support.')