# Import functions, classes, and the streamlit library
import streamlit as st
import os
from scripts.process_resume import TextPipeline
from scripts.employer_postings import get_employer_postings, scrape_indeed_postings

# Define main function
def main():

    # Streamlit title 
    st.title("Find Job Opportunities")

    # Create file uploader with streamlit that accepts one Word doc resume at a time with validation
    uploaded_file = st.file_uploader("Upload your resume - Microsoft Word (.docx) format preferred", type=["docx","PDF"], accept_multiple_files = False)

   # If file is uploaded, process

    if uploaded_file is not None:

        # First, Scrape Resume and Extract Text
        with st.spinner("Processing Resume.."):
            # Initialize TextPipeline class
            txt_pipe = TextPipeline()

            # Extract text from resume
            resume_text = txt_pipe.get_resume_text(uploaded_file)
            #st.write(resume_text)

            # Get person's name
            name = txt_pipe.get_name(resume_text)
            st.write(name)

            # Get person's email
            email = txt_pipe.get_email(resume_text)
            st.write(email)

            # Get person's phone number
            phone = txt_pipe.get_phone(resume_text)
            st.write(phone)

            # CB 7.16 - Commenting out due to API limit
            # Get person's skills
            #skills = txt_pipe.get_skills(resume_text)
            #st.write(skills)

            # CB 7.16 - Commenting out due to inaccuracy
            # Get person's education
            #education = txt_pipe.get_education(resume_text)
            #st.write(education)
        
        # Next, find job matches
        with st.spinner("Finding Matches.."):
            pass

            # Return list of posts as a dataframe and list of URLs to go scrape the web
            # CB 7.16 - Make this run asynchronously and not when a user executes results
            #postings_df, url_list = get_employer_postings()

            # CB 7.16 - Only scraping indeed postings as part of this project. Update to be more comprehensive in future
            #indeed_df = scrape_indeed_postings(url_list)
            #st.dataframe(indeed_df)
        

# Execute main function
if __name__ == "__main__":
    main()