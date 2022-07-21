# Import libraries
import streamlit as st
import plotly.express as px
import pandas as pd
import os
import time

# Import functions and classes
from scripts.process_text import ExtractResumeText, AnalyzeText
from scripts.employer_postings import get_employer_postings, process_URL_postings

# Define main function
def main():
    # Create page config
    st.set_page_config(page_title="Job Connections", page_icon =":handshake:")#,layout="wide")

    # Streamlit title 
    st.title(":handshake: Find Job Opportunities")

    # Initilalize form holder as empty element
    upload_form_holder = st.empty()

    # Initialize Empty Element to hold Screen Content
    screen_content_holder = st.empty()

    # Initialize form to hold file uploader and submit button
    with upload_form_holder.form('file-upload', clear_on_submit=True):
        uploaded_file = st.file_uploader("Upload your resume - Microsoft Word (.docx) format preferred", type=["docx","PDF"], accept_multiple_files = False)
        submitted = st.form_submit_button("Process")

    # Initialize AnalyzeText class
    anlyz_txt = AnalyzeText()

    #### Begin Functionality. Throw error message if form is submitted without a file being uploaded
    if submitted and uploaded_file is None:
        screen_content_holder.error('Please upload a resume')

    # If file is uploaded and form is submitted, process resume
    elif submitted and uploaded_file is not None:

        # Hide form
        upload_form_holder.empty()

        # First, Scrape Resume and Extract Text
        with screen_content_holder.container():

            # First, Scrape Resume and Extract Text
            with st.spinner("Processing resume to find matches..."):
                # Load in job postings existing in data folder to a dataframe
                if os.path.exists("data//postings//Job Postings.xlsx"):
                    jobs_df = pd.read_excel("data//postings//Job Postings.xlsx")
                else:
                    st.error('No jobs found in database. Please update postings')

                ##### Initialize ExtractResumeText class #####
                ext_txt = ExtractResumeText()

                # Extract text from resume
                raw_text = ext_txt.get_resume_text(uploaded_file)
                #st.write(raw_text)

                # Get person's name
                name = ext_txt.get_name(raw_text)
                #st.subheader(name)

                # Get person's email
                email = ext_txt.get_email(raw_text)
                #st.write(email)

                # Get person's phone number
                phone = ext_txt.get_phone(raw_text)
                #st.write(phone)

                # Get person's location
                # CB 7.17 - Update arguments when cleaned up
                resume_location = ext_txt.get_location(raw_text, name)
                #st.write(resume_location)

                # CB 7.16 - Commenting out due to API limit
                # Get person's skills
                #skills = ext_txt.get_skills(raw_text)
                #st.write(skills)
                skills = ''

                # CB 7.16 - Commenting out due to inaccuracy
                # Get person's education
                #education = ext_txt.get_education(raw_text)
                #st.write(education)
                education = ''

                # Append all extracted attributes to a dictionary
                attribute_dict = {'Name':name,'Email':email,'Phone':phone,
                                'Location':resume_location,'Skills':skills,'Education':education}

                ### Build Word Count Features ###
                # Create processed_text by tokening the raw resume text
                processed_text = anlyz_txt.tokenize_text(raw_text)

                ### Analyze word count features and get similarity score from processed_text (Tf-idf and Bag-of-words)
                # Have to pass processed_text in as list as argument expects something iterable
                # Specifying data type as resume
                # Uploaded file = resume
                # Since it is a resume, it returns the resume features
                word_sim_results_df = anlyz_txt.build_and_analyze_word_count_features(data = [processed_text], jobs_df = jobs_df, data_type='resume', name = name)

                ### Build and Analyze Resume Features with Sentence Transformer ###
                sent_trans_sim_df = anlyz_txt.analyze_with_transformer(resume_data = raw_text, jobs_df = jobs_df, data_type='resume')

                # Display success message
                st.success(':smile: Matches found in employer database')
                st.subheader('Best Matches for {}:'.format(name))

                # Join dataframes
                display_df = word_sim_results_df.merge(sent_trans_sim_df, left_on = ['Title'], right_on = ['Title'])

                # Calculate average of tf-idf, BoW, and Sentence Transformer similarities
                display_df['Average Score'] = display_df[['Tf-idf Score', 'BoW Score', 'Transformer Score']].mean(axis = 1)

                # Display dataframe
                st.dataframe(display_df.sort_values(by = ['Average Score'], ascending = False))
               
                #st.dataframe(sent_trans_df.sort_values(by = ['Transformer Score'], ascending = False))

    ######## ---- INITIALIZE SIDEBAR ---- ########
    # Sidebar Image
    st.sidebar.image('data\images\TiO Logo.png')

    # Create Sidebar Title
    st.sidebar.title("Admin Settings")            

    # Create Upload Resumes Button
    if st.sidebar.button('Upload New Resume'):
        screen_content_holder.empty()

    # Create View Job Postings Button
    if st.sidebar.button('View Job Postings'):
        # Hide upload form
        upload_form_holder.empty()

        # Load in job postings existing in data folder to a dataframe
        if os.path.exists("data//postings//Job Postings.xlsx"):
            jobs_df = pd.read_excel("data//postings//Job Postings.xlsx")

            # Replace holder element with container to show current postings
            with screen_content_holder.container():
                st.subheader('Current Job Postings in Database: {}'.format(len(jobs_df)))

                # Display sorted dataframe.. for some reason ascending isn't working correctly
                st.dataframe(jobs_df.loc[:,['Title']].sort_values(by = ['Title'], ascending = True))

        else:
            # Show error message
            screen_content_holder.error('Job data does not exist. Update job postings')

            
    # Create Update Job Postings Button
    if st.sidebar.button('Update Job Postings'):

        # Hide upload form
        upload_form_holder.empty()

        # Return list of posts as a dataframe and list of URLs to go scrape the web
        with screen_content_holder.container():
            with st.spinner("Updating Job Postings..."):

                # Save URLs from employer postings function to be processed
                url_list = get_employer_postings()

            with st.spinner("Scraping job data from URLs..."):

                # CB 7.16 - Only scraping indeed postings as part of this project. Update to be more comprehensive in future
                # Processes and tokenizes URL postings
                jobs_df = process_URL_postings(url_list)

            with st.spinner("Building Word Count Features..."):
                ### Build word count features from processed_text (Tf-idf and Bag-of-words) ###
                # Passing in dataframe as data
                # Specifying data type as jobs
                # Don't have to specify uploaded_file
                anlyz_txt.build_and_analyze_word_count_features(data = jobs_df, data_type='jobs')

            with st.spinner("Creating Semantic Embeddings..."):
                ### Sentence Transformer for jobs ###
                anlyz_txt.analyze_with_transformer(resume_data = "", jobs_df = jobs_df, data_type='jobs')

                # Display success message
                st.success("✔️ Job data has been updated successfully")

                # When processing is finished, show job postings with tokenized text
                st.subheader('Current Job Postings in Database: {}'.format(len(jobs_df)))
                st.dataframe(jobs_df.loc[:,['Title']].sort_values(by = ['Title'], ascending = True))

# Execute main function
if __name__ == "__main__":
    main()