# Connecting Job Seekers to Job Opportunities
 ![image](https://storage.googleapis.com/tio-job-connections-static-images/TiO%20Logo.png)

#### AIPI 540 Deep Learning Applications
#### Project by: Colin Bryan
#### Project Structure: Natural Language Processing
#### Category: Knowledge & Education Management
#### Public-Facing Application: [Link](https://tio-job-connections.ue.r.appspot.com/)

## Background
This is Opportunity (“TiO”) is a social impact company that is focused on breaking down barriers in the job market by empowering people with assets, connections, and confidence.
<br>
<br>
TiO provides professional development and economic advancement solutions for nonprofits to help fill the gaps of their programming and to improve the outcomes for the populations that they serve.

## Problem Statement
TiO's top initiative is a Job Placement Program, where we connect job seekers from our nonprofit partners to employer partners with open positions. Creating a high-quality Job Placement Program is a laborious task due to manual touchpoints and relationship building. 
<br>
<br>
Resumes and job postings need to be collected, documents need to be reviewed, connections need to be facilitated between job seekers and employers, and the list goes on and on.
<br>
<br>
TiO needs a way to efficiently make high-quality connections between job seekers and employers to be able to deliver more impactful results to our customers with our limited workforce. 

## Using the Application
1. Upon accessing the URL, users are asked to upload a resume in **.DOCX** or **.PDF** format

2. After the resume is uploaded, click **Process** to compare the resume to the existing job postings in the database

3. The Top 10 results are displayed to the user based on similarity scores between the resume and the existing job postings. Both word count modeling and transformer modeling are used together to produce an average similarity score for the final output. 

**Note:** *This is an internal tool that will be used by TiO staff. Admin settings will be hidden when this is rolled out for general use by partner organizations.*

## Getting Started
---------------
1. To run locally: Clone the repository, create a virtual environment, and install the requirements needed to run the application
```
pip install -r requirements.txt
```
2. Start the Streamlit app
```
streamlit run main.py

3. Job postings are maintained in an Excel file in Google Cloud Storage. Non-TiO users will not be able to update this file.

4. Modeling approaches can be found and edited in the AnalyzeText() class.
```

## Future Work
* Incorporating additional attributes and data sources
* Testing new models and scoring metrics
* Building out ancillary capabilities on top of this platform


## Citation
The following source was used to guide my approach to extract attributes from text. Although the final application codebase looks different from the contents contained in this blog, this post was instrumental in serving as a starting point.
<br>
<br>
[1] V. Angelova, [Build your own Resume Parser Using Python and NLP](https://blog.apilayer.com/build-your-own-resume-parser-using-python-and-nlp/)

```
@InProceedings{Build your own Resume Parser Using Python and NLP,
  author    = {Victoria Angelova,
  title     = {Build your own Resume Parser Using Python and NLP},
  year      = {2022},
  publisher = {APILayer},
  url       = {(https://blog.apilayer.com/build-your-own-resume-parser-using-python-and-nlp/)}
}
```
