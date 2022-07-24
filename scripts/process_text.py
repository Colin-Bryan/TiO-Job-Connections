# Import functions
from scripts.api import skills_API

## Import libraries
import pandas as pd
import numpy as np
import re
import requests
import string
from datetime import datetime
from pathlib import Path
import pickle
import shutil
import json
import io

# .pdf processing
from pdfminer.high_level import extract_text
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO

# .docx processing
import docx2txt
import textract

# NLP processing libraries and modules
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

from sentence_transformers import SentenceTransformer, util

import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('vader_lexicon')


#Create class for extracting text
class ExtractResumeText():
    '''
    A class to extract text from uploaded resumes.

    ...

    Attributes
    ----------
    None

    Methods
    -------
    get_resume_text(self, document):
        Extracts the raw text from the resume depending on file format.

    get_name(self, raw_text):
        Gets the job seeker's name from the resume.

    get_phone(self, raw_text):
        Gets the job seeker's phone number from the resume.

    get_location(self, raw_text, name):
        Gets the job seeker's location from the resume.

    get_skills(self, raw_text):
        Gets the job seeker's skills from the resume by comparing to the Skills Database from API Layer.
    '''
    
    # Initialize class
    def __init__(self):
        '''
        No attributes to initialize.
        '''
        return

    # Extract text from resume
    def get_resume_text(self, document):
        '''
        Returns the raw text from the resume depending on document type.

        Parameters:
            document (file): The uploaded resume.
        
        Returns:
            raw_text (str): The resume text in string format.
        '''
        
        # Use try-excepts to process .docx first
        try: 
            # Use doc2txt's process function to get resume text from .docx
            raw_text = docx2txt.process(document)

        # If not .docx, try pdf
        # CB 7.17 Note: pdf scraping is not exactly a 1:1 to docx scraping so doc type might impact results currently
        except:
            try:
                # Initialize device from pdfminer to get resume text from pdf
                device = TextConverter(PDFResourceManager(), StringIO, laparams=LAParams())
                
                # Initialize interpreter and get file_pages
                interpreter = PDFPageInterpreter(PDFResourceManager(), device)
                file_pages = PDFPage.get_pages(document)
                
                # Get raw text
                raw_text = extract_text(document)
            except:
                # CB 7.16 - Commenting out because we will not be accepting .doc
                # Use textract's process function to get resume text from .doc
                #raw_text = textract.process(document)       
                st.error('Text cannot be extracted from the resume. Please review the file and try again.')

        return raw_text.replace('\t', ' ')
    
    def get_name(self, raw_text):
        '''
        Returns the first line of the resume as the person's name.

        Parameters:
            raw_text (str): The raw text from the resume.
        
        Returns:
            raw_text.split('\n')[0] (str): The first line of the resume.
        '''

        # Placeholder to just get the first line from the resume and assume it's the first name.
        # CB 7.24 - Will be cleaned up in v2
        return raw_text.split('\n')[0]


    def get_email(self, raw_text):
        '''
        Returns the job seeker's email address in the resume if it exists.

        Parameters:
            raw_text (str): The raw text from the resume.
        
        Returns:
            Either the first email address found in the resume of a string that says that the email could not be located."
        '''
        try:
            # Use regular expression to find email
            email_reg = re.compile(r'\S+@\S+')

            # Return first email found for the person
            return re.findall(email_reg, raw_text)[0]

        except:
            return 'Could not find email'

    def get_phone(self, raw_text):
        '''
        Returns the job seeker's phone number in the resume if it exists.

        Parameters:
            raw_text (str): The raw text from the resume.
        
        Returns:
            Either the first phone number found in the resume of a string that says that the phone number could not be located."
        '''
        try: 
            # Use regular expression to find phone
            phone_reg = re.compile(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]')

            # Return first phone number found for the person
            return re.findall(phone_reg, raw_text)[0]
        
        except:
            return 'Could not find phone number'

    def get_location(self, raw_text, name):
        '''
        Returns the job seeker's location in the resume if it exists.

        Parameters:
            raw_text (str): The raw text from the resume.
        
        Returns:
            Either the job seeker's location or a string that says that the location could not be located."
        '''

        try: 
            # Use regular expression to find location
            location_reg = re.search(r'([^\d]+)?(\d{5})?', raw_text).group(0)
            
            ### CB 7.16 - Come back and clean up the following ###
            # Don't have a good way to replace trailing punctuation yet
            location_reg = location_reg.replace('•','')
            #Replace name that is coming back in match and strip out white spaces
            location_reg = location_reg.replace(name,'').strip()

            # Return location if it ends with a digit 
            if location_reg[-1].isdigit():
                return location_reg
            
            # Else split on a new line and return the first part
            else:
                return location_reg.split('\n')[0]

        except:
            return 'Could not find location'
            

    def get_skills(self, raw_text):
        '''
        Returns the job seeker's skills in the resume by comparing to the Skills database from API Layer. 
        Boilerplate code from 'https://blog.apilayer.com/build-your-own-resume-parser-using-python-and-nlp/'

        Parameters:
            raw_text (str): The raw text from the resume.
        
        Returns:
            found_skills (list): A list of found skills from the Skills Database from API Layer.
        '''
        stop_words = set(nltk.corpus.stopwords.words('english'))
        word_tokens = nltk.tokenize.word_tokenize(raw_text)
    
        # remove the stop words
        filtered_tokens = [w for w in word_tokens if w not in stop_words]
    
        # remove the punctuation
        filtered_tokens = [w for w in word_tokens if w.isalpha()]
    
        # generate bigrams and trigrams (such as artificial intelligence)
        bigrams_trigrams = list(map(' '.join, nltk.everygrams(filtered_tokens, 2, 3)))
    
        # we create a set to keep the results in.
        found_skills = set()
    
        # we search for each token in our skills database
        for token in filtered_tokens:
            if skills_API(token.lower()):
                found_skills.add(token)
    
        # we search for each bigram and trigram in our skills database
        for ngram in bigrams_trigrams:
            if skills_API(ngram.lower()):
                found_skills.add(ngram)
    
        return found_skills


class AnalyzeText():
    '''
    A class to analyze text and process text in modeling.

    ...

    Attributes
    ----------
    None

    Methods
    -------
    tokenize_text(self, raw_text):
        Tokenizes raw text for use in modeling.

    build_and_analyze_word_count_features(self, data, jobs_df = 'Placeholder', data_type='resume'):
        Builds and analyzes text for word count modeling.

    analyze_with_transformer(self, resume_data, jobs_df, data_type, name = 'Pete Fagan'): 
        Analyzes text using transformer models.
    
    output_to_JSON(self, name, final_postings_df):
        Outputs a DataFrame to JSON format.
    '''
    # Initialize class
    def __init__(self):
        '''
        No attributes to initialize.
        '''
        return
    
    # Tokenize, remove stop words and punctuation, lemmatize
    def tokenize_text(self, raw_text):
        '''
        Returns tokenized text that is lemmatized with stop words and punctuation removed.

        Parameters:
            raw_text (str): The raw text from the resume or job postings.
        
        Returns:
            A string of tokens joined together by spaces.
        '''

        # Create stop words and punctuation
        stop_words = set(nltk.corpus.stopwords.words('english'))
        punctuation = string.punctuation

        # Load in user defined stop words and save to list
        user_defined_stop_words = pd.read_csv('data//stop_words//custom stop words.csv')
        user_defined_stop_words = list(user_defined_stop_words)

        # Define punctuation to remove in addition to string.punctuation
        user_defined_punctuation = ['–','•','’']

        # Create word_tokens from resume
        tokens = nltk.tokenize.word_tokenize(raw_text)
    
        # remove the stop words, string.punctuation, and user_defined_punctuation
        tokens = [w for w in tokens if w not in stop_words and w not in punctuation
                 and w not in user_defined_punctuation and w not in user_defined_stop_words]
    
        # Lemmatize text
        wordnet_lemmatizer = WordNetLemmatizer()
        tokens = [wordnet_lemmatizer.lemmatize(word).lower().strip() for word in tokens]

        # Return processed text joined together
        return " ".join([i for i in tokens])
    
    # Create features from processed tokens (takes data, type of data being passed - default is resume)
    # CB 7.17 - Might have to strip out into 2 separate functions
    def build_and_analyze_word_count_features(self, data, jobs_df = 'Placeholder', data_type='resume'):
        '''
        Builds features from either resumes or job postings and performs word count modeling.

        Parameters:
            data : An argument for the data to be modeled. This is treated differently depending on the data_type argument.
            jobs_df : An argument that accepts the jobs_df DataFrame if it is needed (not required for resumes).
            data_type (str) : A string specifiying if a 'resume' or 'jobs' are being processed (default = resume).
        
        Returns:
            If a resume is being processed, a DataFrame of cosine_similarity scores from the job seeker's resume
            to each job posting is returned for Tf-idf and Bag-of-words modeling methods.

            If job postings are being processed, Tf-idf and Count vectorizers, plus resulting matrices, are saved as 
            .pkl files for use in resume comparison. Nothing is returned.
        '''
        # If data being passed in for jobs, build features
        if data_type == 'jobs':

            # Conduct TFIDF Vectorization and Count Vectorization using uniqrams and bigrams
            tfidf_vec = TfidfVectorizer(ngram_range = (1,2))
            count_vec = CountVectorizer(ngram_range = (1,2))

            # Apply fit_transform to data from vectorizers to train them
            tfidf_postings_result = tfidf_vec.fit_transform(data)
            count_postings_result = count_vec.fit_transform(data)

            ### Save Tf-idf files ###
            # Save tfidf vectorizer to pickle file at specified path
            path = Path('data//vectorizers/tfidf_vectorizer.pkl')
            with path.open('wb') as fp:
                pickle.dump(tfidf_vec, fp)

            # Save tfidf result to pickle file at specified path
            path = Path('data//word_count_matrices/tfidf_matrix.pkl')
            with path.open('wb') as fp:
                pickle.dump(tfidf_postings_result, fp)
            
            ### Save Count files ###
            # Save count vectorizer to pickle file at specified path
            path = Path('data//vectorizers/count_vectorizer.pkl')
            with path.open('wb') as fp:
                pickle.dump(count_vec, fp)

            # Save count result to pickle file at specified path
            path = Path('data//word_count_matrices/count_matrix.pkl')
            with path.open('wb') as fp:
                pickle.dump(count_postings_result, fp)
            
                
        # Else, process resumes
        else: 
            
            ### Load Tf-idf files ###
            # Load in pickle file for tfidf vectorizer
            with open('data//vectorizers//tfidf_vectorizer.pkl', 'rb') as file:
                tfidf_vec = pickle.load(file)

            # Load in pickle file for pre-trained tfidf matrix
            with open('data//word_count_matrices/tfidf_matrix.pkl', 'rb') as file:
                tfidf_postings_result = pickle.load(file)

            ### Load Count files ###
            # Load in pickle file for count vectorizer
            with open('data//vectorizers//count_vectorizer.pkl', 'rb') as file:
                count_vec = pickle.load(file)

            # Load in pickle file for pre-trained count matrix
            with open('data//word_count_matrices/count_matrix.pkl', 'rb') as file:
                count_postings_result = pickle.load(file)

            # Apply transform to resume data from pre-trained vectorizers
            tfidf_resume_result = tfidf_vec.transform(data)
            count_resume_result = count_vec.transform(data)

            # Initialize empty lists for title and similarity scores
            title_list = []
            tfidf_sim_list = []
            count_sim_list = []

            #CB 7.20 - Can probably put this into 2 different functions in find_matches.py:
            ### TFIDF Similarity ###
            # Loop through each similarity result for job postings
            for i, vector in enumerate(tfidf_postings_result):

                # Get the job posting title
                title = jobs_df.loc[i,'Title']
                # Get the resume to job posting similarity score
                tfidf_cosine_similarity = cosine_similarity(tfidf_resume_result,vector)[0][0]

                # Append variables to list and loop again
                title_list.append(title)
                tfidf_sim_list.append(tfidf_cosine_similarity)
                
            #Create dictionary of Job Titles to Cosine Similarity Scores
            tfidf_dict = dict(zip(title_list, tfidf_sim_list))
 
            ### BoW Similarity ###
            # Loop through each similarity result for job postings
            for i, vector in enumerate(count_postings_result):

                # Get the job posting title
                title = jobs_df.iloc[i,0]
                
                # Get the resume to job posting similarity score
                count_cosine_similarity = cosine_similarity(count_resume_result,vector)[0][0]

                # Append variables to list and loop again
                title_list.append(title)
                count_sim_list.append(count_cosine_similarity)
            
            #Create dictionary of Job Titles to Cosine Similarity Scores
            count_dict = dict(zip(title_list, count_sim_list))

            # Put dicts into dataframe
            tfidf_df = pd.DataFrame(tfidf_dict.items(), columns = ['Title','Tf-idf Score'])
            count_df = pd.DataFrame(count_dict.items(), columns = ['Title', 'BoW Score'])

            #Join dfs on "Title"
            results_df = tfidf_df.merge(count_df, left_on=['Title'], right_on =['Title'])

            # Create left dataframe to display results with Employer, Title, and URL
            left_df = jobs_df.loc[:,['Employer', 'Title', 'URL']]

            # Create final returned df
            returned_df = left_df.merge(results_df, left_on=['Title'], right_on =['Title'])

            # Return dataframe
            return returned_df
    
    def analyze_with_transformer(self, resume_data, jobs_df, data_type, name = 'Pete Fagan'): 
        '''
        Analyzes text from either resumes or job postings and performs transformer modeling.

        Parameters:
            resume_data (str): The raw text form the resume is passed in for resume processing. This is not required for job posting modeling.
            jobs_df (DataFrame): The jobs_df DataFrame.
            data_type (str) : A string specifiying if a 'resume' or 'jobs' are being processed (default = resume).
            name (str): The name of the job seeker. Pete Fagan is defaulted because it is not required for jobs.

        Returns:
            If a resume is being processed, a DataFrame of cosine_similarity scores from the job seeker's resume
            to each job posting is returned for transformer modeling methods.

            If job postings are being processed, transformer embeddings are saved as a
            .pkl file for use in resume comparison. Nothing is returned.
        '''
        # Load in MiniLM-L6-v2 transformer model pre-trained
        sent_trans_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        # Analyze jobs
        if data_type == 'jobs':

            # Create embeddings for text from full_text already created (don't need to tokenize/remove stop words/lemmatize)
            job_transformed = jobs_df['full_text'].values.tolist()
            embeddings = [sent_trans_model.encode(job) for job in job_transformed]

            # Save embeddings as pickle file at specified path to use in application
            path = Path('data//embeddings//job_embeddings.pkl')
            with path.open('wb') as fp:
                pickle.dump(embeddings, fp)

            # Save embeddings as pickle file at specified path to archive
            path = Path('data//embeddings//archive//job_embeddings_{}.pkl'.format(datetime.now().strftime("%Y-%m-%d")))
            with path.open('wb') as fp:
                pickle.dump(embeddings, fp)

        # Analyze resume
        else:

            # Initialize lists to hold values
            title_list = []
            sent_trans_sim_list = []

            # Load job embeddings from pickle
            with open('data//embeddings//job_embeddings.pkl', 'rb') as file:
                job_embeddings = pickle.load(file)

            # Pass in raw_text from resume ("data") into list and assign to variable
            resume_to_list = [resume_data]
            resume_embeddings = sent_trans_model.encode(resume_to_list) #for resume in resume_transformed

            # Calculate Embedding Similarity
            # Loop through each job_embedding
            for i, vector in enumerate(job_embeddings):

                # Get the job posting title
                title = jobs_df.loc[i,'Title']
                
                # Get the resume to job posting similarity score (have to detach values and convert to np floats)
                sent_trans_cosine_similarity = util.cos_sim(resume_embeddings,vector)[0][0].detach().numpy()

                # Append variables to list and loop again
                title_list.append(title)
                sent_trans_sim_list.append(float(sent_trans_cosine_similarity))
            
            #Create dictionary of Job Titles to Cosine Similarity Scores
            sent_trans_dict = dict(zip(title_list, sent_trans_sim_list))

            # Put dicts into dataframe
            sent_trans_df = pd.DataFrame(sent_trans_dict.items(), columns = ['Title','Transformer Score'])

            #Return dataframe
            return sent_trans_df

    def output_to_JSON(self, name, final_postings_df):
        '''
        Outputs the final DataFrame of similarity scores to job postings for the user to JSON format 
        as a list of JSON objects.

        Parameters:
            name (str): The name of the job seeker.
            final_postings_df (DataFrame): The final DataFrame containing Tf-idf, Bag-of-words, and Transformer similarity scores
                                            for each job posting to the job seeker's resume.

        Returns:
            Nothing. A JSON file is outputted for the job seeker.
        '''
        
        # Initialize list to store JSON objects
        json_list = []

        # Loop through final postings dataframe for the user and put data in dictionary format
        for i in range(len(final_postings_df)):

            # Append each dictionary to the list
            json_list.append(
                {"Employer":final_postings_df.loc[i,"Employer"].replace('\xa0',''),
                "Title":final_postings_df.loc[i,"Title"],
                "URL":final_postings_df.loc[i,"URL"],
                "Score":final_postings_df.loc[i,"Average Score"]}
            )

        # Write JSON file withthe list
        with io.open('data\\outputs\\{}_output.json'.format(name), 'w', encoding='utf8') as outfile:
            json_data = json.dumps(json_list,
                            indent=4, sort_keys=True,
                            separators=(',', ': '), ensure_ascii=False)
            outfile.write(json_data)
