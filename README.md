# Connecting Job Seekers to Job Opportunities
 ![image](https://storage.googleapis.com/tio-job-connections-static-images/TiO%20Logo.png)

#### AIPI 540 Deep Learning Applications
#### Project by: Colin Bryan
#### Project Structure: Natural Language Processing
#### Category: Knowledge & Education Management
#### Public-Facing Application: [Link](https://tio-job-connections.ue.r.appspot.com/)

Background
----------
This is Opportunity (“TiO”) is a social impact company that is focused on breaking down barriers in the job market by empowering people with assets, connections, and confidence.
<br>
<br>
TiO provides professional development and economic advancement solutions for nonprofits to help fill the gaps of their programming and to improve the outcomes for the populations that they serve.

Problem Statement
-----------------
* TiO's top initiative is a Job Placement Program, where we connect job seekers from our nonprofit partners to employer partners with open positions. 
* Creating a high-quality Job Placement Program is a laborious task due to manual touchpoints and relationship building. 
* Resumes and job postings need to be collected, documents need to be reviewed, connections need to be facilitated between job seekers and employers, and the list goes on and on.
* TiO needs a way to efficiently make high-quality connections between job seekers and employers to be able to deliver more impactful results to our customers. 

Using the Application
---------------------


Getting Started
---------------
1. To run locally: Clone the repository, create a virtual environment, and install the requirements needed to run the application
```
pip install -r requirements.txt
```
2. Download the nltk modules required to run the application by following these steps:
```
python 
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('vader_lexicon')
```
3. Start the Streamlit app
```
streamlit run main.py
```

Data Sourcing, Processing, & Modeling
-------------------------------------
* Data is sourced from the Reddit API when a user enters a stock's ticker on the Streamlit User Interface in the following steps:

     1) API queries Reddit for the search term
    
     2) Retrieves the 100 most recent posts related to the search term and returns posts categorized as "Company           News" or "News" as a dataframe
     
     3) If "News" posts aren't found, the user is prompted to enter a new search term. If "News" posts are found,         the following steps happen under the hood:
     
           * The Reddit posts are put through a data pipeline extracting phrases from the title and content of                  the posts. This process is called "chunking" 
           
           * From analyzing all of the returned posts by passing the text chunks through a generative summary                  model that summary is rendered onto the User Interface 
           
           * Sentiment analysis is conducted by lemmatizing the Reddit posts and returning a list of lemmatized                words from the posts. This list is then compared to the Loughran-McDonald Sentiment Word List, a                  popular finance word-to-sentiment mapping repository, to understand the frequency of sentiment-                    carrying words that occur in each Reddit post that is processed
           
           * A count of the frequency of sentiment-carrying words in the Reddit posts for the queried stock                    ticker is conducted and used to make a recommendation on buying, selling, or waiting on the stock. A              graph of sentiment-carrying word frequency is also displayed through the User Interface

Adavantages of Deep Learning vs. Non-Deep Learning in Finance
---------------
NLP is better at creating structure out of unstructured data and recognizing finance jargon

NLP enriches decision-making in the following ways:

   1) Automation: Replace manual process or turning unstructured data into a more usable form i.e. automating capture of earnings calls
    
   2) Data Enrichment: Add context to captured, unstructured data to make it more searchable and actionable i.e. searching for a particular topic in a financial filing
     
   3) Search and Discovery: Create a competitive advantage from a variety of sources
     
      (Source: MIT Sloan School of Management)



Model Evaluation & Results
----------------------------

* The tool works best for popular stocks on Reddit
* For stocks with less trading volume, "News" posts often do not exist or the ticker is mistaken for a different topic on Reddit
* For best results, search for more popular assets, like blue-chip stocks

Note: the software platform makes use of the terms, bullish and bearish
* 🐂 Bullish: Postive sentiment in which investors believe a stock or the broader market will appreciate in value
* 🐻 Bearish: Negative sentiment in which investors believe a stock or the broader market will depreciate in   value


Streamlit Demonstration on Coinbase (Ticker: COIN)
<br>
<br>
Ticker Search:

https://user-images.githubusercontent.com/78511177/180041852-69e2c8a4-5031-49f7-97a6-21f5d6bc673e.mp4

Summary Generation:



https://user-images.githubusercontent.com/78511177/180041893-c5568833-a517-4893-b632-5d11305b42e0.mp4



Results Based on Sentiment Analysis:

Suggested Buy Signal


<img width="434" alt="Screen Shot 2022-07-19 at 6 14 19 PM" src="https://user-images.githubusercontent.com/78511177/179857576-74286598-141e-42b3-8d66-b09eeb82ded7.png">

<br>

Suggested Sell Signal


<img width="432" alt="Screen Shot 2022-07-19 at 6 16 44 PM" src="https://user-images.githubusercontent.com/78511177/179857811-b1fa053f-c93e-4ed6-a252-1ba75c7fe814.png">









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
