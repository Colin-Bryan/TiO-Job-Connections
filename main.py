# Import functions, classes, and the streamlit library
import streamlit as st
import plotly.express as px
import pandas as pd
import os
import time
from scripts.process_resume import ExtractText, AnalyzeText
from scripts.employer_postings import get_employer_postings, scrape_indeed_postings


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

    # Throw error message if form is submitted without a file being uploaded
    if submitted and uploaded_file is None:
        screen_content_holder.error('Please upload a resume')

    # If file is uploaded and form is submitted, process resume
    elif submitted and uploaded_file is not None:

        # Hide form
        upload_form_holder.empty()

        # First, Scrape Resume and Extract Text
        with screen_content_holder.container():

            # First, Scrape Resume and Extract Text
            with st.spinner("Processing Resume.."):
                ##### Initialize ExtractText class #####
                ext_txt = ExtractText()

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

                ##### Initialize AnalyzeText class #####
                anlyz_txt = AnalyzeText()

                # Create processed_text by tokening the raw resume text
                processed_text = anlyz_txt.tokenize_text(raw_text)

                #  Build features from processed_text
                df_list = anlyz_txt.word_count_features(processed_text)

                # Display DataFrames by looping
                for i, word_count_df in enumerate(df_list):
                    if i == 0:
                        st.subheader('Unigrams Only')
                    if i == 1:
                        st.subheader('Bigrams Only')
                    elif i == 2:
                        st.subheader('Trigrams Only')

                    # Write top 10 results 
                    st.dataframe(word_count_df.sort_values(by=['Tf-idf Score'], ascending = False)[0:10])

                # Chunk text
                #txt, attrdict = anlyz_txt.chunk_text(raw_text, attribute_dict)

            # Next, find job matches
            with st.spinner("Finding Matches.."):

                # Load in job postings existing in data folder to a dataframe
                if os.path.exists("data//Job Postings.xlsx"):
                    jobs_df = pd.read_excel("data/Job Postings.xlsx")
                    #st.dataframe(jobs_df[['Title','URL']])
                else:
                    st.error('No job postings found for comparison')

        
        ### CB 7.16 - Commenting out filter functionality
        # # Create multiselect filter for location
        # st.subheader('Filter Results')
        # location = st.multiselect("Choose Location",
        #     options=jobs_df["Location"].unique(),
        #     default=jobs_df["Location"].unique()
        # )

        # # Create query for displaying matching jobs
        # df_selection = jobs_df.query(
        #     "Location == @location"
        # )

        # Display Job Title and URL on page
        #st.dataframe(df_selection[['Title','URL']])



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
        if os.path.exists("data//Job Postings.xlsx"):
            jobs_df = pd.read_excel("data/Job Postings.xlsx")

            # Replace holder element with container to show current postings
            with screen_content_holder.container():
                st.subheader('Current Job Postings in Database:')
                st.dataframe(jobs_df)

        else:
            # Show error message
            screen_content_holder.error('Job data does not exist. Update job postings')

            # Show job postings
        #     st.subheader('Current Job Postings in Database:')
        #     st.dataframe(jobs_df)
            
        # else:
        #     st.error('Job data does not exist. Update job postings')
            
    # Create Update Job Postings Button
    if st.sidebar.button('Update Job Postings'):

        # Hide upload form
        upload_form_holder.empty()

        # Return list of posts as a dataframe and list of URLs to go scrape the web
        with screen_content_holder.container():
            with st.spinner("Updating Job Postings.."):

                # Save URLs from employer postings function to be processed
                url_list = get_employer_postings()

                # CB 7.16 - Only scraping indeed postings as part of this project. Update to be more comprehensive in future
                scrape_indeed_postings(url_list)

                # Display success message
                st.success("✔️ Job data has been updated successfully")

                if os.path.exists("data//Job Postings.xlsx"):
                    jobs_df = pd.read_excel("data/Job Postings.xlsx")
        
                    # Show job postings
                    st.subheader('Current Job Postings in Database:')
                    st.dataframe(jobs_df)

    
    # #### ---- INITIALIZE SIDEBAR ---- ####
    # st.sidebar.title("Admin Settings")            

    # if st.sidebar.button('Upload Resumes'):
    #     pass

    # # When view job postings button is clicked, display existing data
    # if st.sidebar.button('View Job Postings'):

    #     # Load in job postings existing in data folder to a dataframe
    #     if os.path.exists("data//Job Postings.xlsx"):
    #         jobs_df = pd.read_excel("data/Job Postings.xlsx")

    #         # Replace holder element with container to show current postings
    #         with fileuploader_holder.container():
    #             st.subheader('Current Job Postings in Database:')
    #             st.dataframe(jobs_df)

    #     else:
    #         # Show error message
    #         fileuploader_holder.error('Job data does not exist. Update job postings')

    #         # Show job postings
    #     #     st.subheader('Current Job Postings in Database:')
    #     #     st.dataframe(jobs_df)
            
    #     # else:
    #     #     st.error('Job data does not exist. Update job postings')
            
    # # When Update Job Postings button is clicked, update data
    # if st.sidebar.button('Update Job Postings'):

    #     # Remove file uploader temporarily
    #     fileuploader_holder.empty()

    #     # Return list of posts as a dataframe and list of URLs to go scrape the web
    #     with fileuploader_holder.container():
    #         with st.spinner("Updating Job Postings.."):

    #             # Save URLs from employer postings function to be processed
    #             url_list = get_employer_postings()

    #             # CB 7.16 - Only scraping indeed postings as part of this project. Update to be more comprehensive in future
    #             scrape_indeed_postings(url_list)

    #             # Display success message
    #             st.success("✔️ Job data has been updated successfully")

    #             if os.path.exists("data//Job Postings.xlsx"):
    #                 jobs_df = pd.read_excel("data/Job Postings.xlsx")
        
    #                 # Show job postings
    #                 st.subheader('Current Job Postings in Database:')
    #                 st.dataframe(jobs_df)

        # CB 7.16 - Commenting out. Use as reference
        # with st.empty():
        #     for seconds in range(60):
        #         st.success("✔️ Job data has been updated successfully")
        #         time.sleep(0.75)
        #     st.write("")

    # ---- END SIDEBAR

    # # If file is uploaded, process
    # if uploaded_file is not None:

    #     # First, Scrape Resume and Extract Text
    #     with st.spinner("Processing Resume.."):
    #         # Initialize ExtractText class
    #         ext_txt = ExtractText()

    #         # Extract text from resume
    #         raw_text = ext_txt.get_raw_text(uploaded_file)
    #         #st.write(raw_text)

    #         # Get person's name
    #         name = ext_txt.get_name(raw_text)
    #         st.write(name)

    #         # Get person's email
    #         email = ext_txt.get_email(raw_text)
    #         st.write(email)

    #         # Get person's phone number
    #         phone = ext_txt.get_phone(raw_text)
    #         st.write(phone)

    #         # CB 7.16 - Commenting out due to API limit
    #         # Get person's skills
    #         #skills = ext_txt.get_skills(raw_text)
    #         #st.write(skills)

    #         # CB 7.16 - Commenting out due to inaccuracy
    #         # Get person's education
    #         #education = ext_txt.get_education(raw_text)
    #         #st.write(education)
        
    #     # Next, find job matches
    #     with st.spinner("Finding Matches.."):
    #         # Load in job postings existing in data folder to a dataframe
    #         if os.path.exists("data//Job Postings.xlsx"):
    #             jobs_df = pd.read_excel("data/Job Postings.xlsx")

            
    #     # --- UPDATE SIDEBAR AFTER PROCESSING ---- 

    #     # Create multiselect filter for location
    #     location = st.sidebar.multiselect(
    #         "Select the Location:",
    #         options=jobs_df["Location"].unique(),
    #         default=jobs_df["Location"].unique()
    #     )

    #     # Create query for displaying matching jobs
    #     df_selection = jobs_df.query(
    #         "Location == @location"
    #     )

    #     # Display Job Title and URL on page
    #     st.text("")
    #     st.subheader("Top Results")
    #     st.dataframe(df_selection[['Title','URL']])

# Execute main function
if __name__ == "__main__":
    main()