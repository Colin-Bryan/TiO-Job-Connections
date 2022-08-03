# Connecting Job Seekers to Job Opportunities
 ![image](https://storage.googleapis.com/tio-job-connections-static-images/TiO%20Logo.png)

#### AIPI 540 Deep Learning Applications
#### Project by: Colin Bryan
#### Project Structure: Natural Language Processing
#### Category: Knowledge & Education Management
#### Public-Facing Application: [Link](https://tio-job-connections.ue.r.appspot.com/)

## Background
### This is Opportunity (“TiO”) is a social impact company that is focused on breaking down barriers in the job market by empowering people with assets, connections, and confidence.
<br>
<br>
### TiO provides professional development and economic advancement solutions for nonprofits to help fill the gaps of their programming and to improve the outcomes for the populations that they serve.

## Problem Statement
### TiO's top initiative is a Job Placement Program, where we connect job seekers from our nonprofit partners to employer partners with open positions. Creating a high-quality Job Placement Program is a laborious task due to manual touchpoints and relationship building. 
<br>
<br>
### Resumes and job postings need to be collected, documents need to be reviewed, connections need to be facilitated between job seekers and employers, and the list goes on and on.
<br>
<br>
### TiO needs a way to efficiently make high-quality connections between job seekers and employers to be able to deliver more impactful results to our customers with our limited workforce. 

## Using the Application
1. Upon accessing the URL, users are asked to upload a resume in **.DOCX** or **.PDF** format

2. After the resume is uploaded, click **Process** to compare the resume to the existing job postings in the database

3. The Top 10 results are displayed to the user based on similarity scores between the resume and the existing job postings

**Note:** *This is an internal tool that will be used by TiO staff. Admin settings will be hidden when this is rolled out for general use by partner organizations.*

### Getting Started
---------------
1. To run locally: Clone the repository, create a virtual environment, and install the requirements needed to run the application
```
pip install -r requirements.txt
```
2. Start the Streamlit app
```
streamlit run main.py

3. Job postings are maintained in an Excel file in Google Cloud Storage. Non-TiO users will not be able to update this file.
```


Future Work
------------
* Improve the query passed to the Reddit API to ensure only stock-related posts are returned
* Generate more posts for both popular and less popular tickers
* Apply weights to the posts based on the subreddit where the post is found (e.g., r/wallstreetbets, r/stocks, r/superstonk, r/personalfinance, r/investing, etc.)
* Expand support for alternative asset classes, like cryptocurrencies and commodities
* Enable cross-platform support for phone and tablet
* Allow for scale: multiple users across devices should simultaneously be able to generate results


Conclusion
----------
"Be fearful when others are greedy. Be greedy when others are fearful." - Warren Buffett
* Being non-concensus in the way one invests is the only way to generate superior returns 
* Using non-traditional data sources, like Reddit, have come into vogue and came help all types of investors glean insights they would not otherwise have readily available
* Coupling technology and discipline, retail investors can level the playing field and even beat financial institutions at their own game

Additional Resources
--------------------
* Corresponding Slide Deck: [here](https://prodduke-my.sharepoint.com/:p:/g/personal/da204_duke_edu/Ec1ZDUDSbthGunjGt8mpvN4BtHJSYaoSMXipR13o9brTjA?e=vf53j4)

* Financial Sentiment Analysis - Project by Prosus: [here](https://github.com/ProsusAI/finBERT)

* NLP in the Stock Market - Project by Roshan Adusumilli: [here](https://towardsdatascience.com/nlp-in-the-stock-market-8760d062eb92#:~:text=Machine%20learning%20models%20implemented%20in,forms%20to%20forecast%20stock%20movements.)

* Loughran McDonald-SA-2020 Sentiment Word List: [here](https://researchdata.up.ac.za/articles/dataset/Loughran_McDonald-SA-2020_Sentiment_Word_List/14401178)

* Famed Investor Howard Marks' Memo on Non-Consensus Investing: [here](https://www.oaktreecapital.com/docs/default-source/memos/1993-02-15-the-value-of-predictions-or-where-39-d-all-this-rain-come-from.pdf?sfvrsn=6fbc0f65_2)


Citation
--------
[1] R. Adusumilli, [NLP in the Stock Market](/https://github.com/roshan-adusumilli/nlp_10-ks)

```
@InProceedings{NLP_on_Financial_Statements,
  author    = {Roshan Adusumilli},
  title     = {NLP in the Stock Market},
  year      = {2020},
  publisher = {GitHub},
  journal   = {GitHub repository},
  url       = {https://github.com/roshan-adusumilli/nlp_10-ks}
}
```
