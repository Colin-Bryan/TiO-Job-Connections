# Import funcitons
from scripts.api import skills_API

## Import libraries
import pandas as pd
import numpy as np
import re
import requests
import string

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


#Create class for text pipeline
class ExtractText():
    # Initialize class
    def __init__(self):
        return

    # Extract text from resume
    def get_resume_text(self, document):
        
        # Use try-excepts to process .docx first
        try: 
            # Use doc2txt's process function to get resume text from .docx
            raw_text = docx2txt.process(document)

        # If not .docx, try pdf
        except:
            try:
                # Use pdfminer to get resume text from pdf
                # CB 7.16 - Can be improved to clean up formatting

                # Initialize device from pdfminer to get resume text from pdf
                device = TextConverter(PDFResourceManager(), StringIO, laparams=LAParams())
                
                # Initialize interpreter and get file_pages
                interpreter = PDFPageInterpreter(PDFResourceManager(), device)
                file_pages = PDFPage.get_pages(document)
                
                raw_text = extract_text(document)
            except:
                # CB 7.16 - Commenting out because we will not be accepting .doc
                # Use textract's process function to get resume text from .doc
                #raw_text = textract.process(document)       
                pass

        return raw_text.replace('\t', ' ')
    
    # CB 7.16 - Extract name - need to fix intelligence. Come back and clean up
    def get_name(self, raw_text):

        # for sent in nltk.sent_tokenize(raw_text):
        #     for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
        #         if hasattr(chunk, 'label') and chunk.label() == 'PERSON':
        #             person_name =' '.join(chunk_leave[0] for chunk_leave in chunk.leaves())
        
        # Placeholder to just get the first line from the resume and assume it's the first name
        return raw_text.split('\n')[0]

    # Extract email
    def get_email(self, raw_text):
        
        # Use regular expression to find email
        email_reg = re.compile(r'[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+')

        # Return first email found for the person
        return re.findall(email_reg, raw_text)[0]

    # Extract phone number
    def get_phone(self, raw_text):
        
        # Use regular expression to find phone
        phone_reg = re.compile(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]')
        
        # Return first phone number found for the person
        return re.findall(phone_reg, raw_text)[0]

    # Get the candidate's location. CB 7.17 - This is very hacky, need to replace functionality for getting location
    def get_location(self, raw_text, name):

        # Use regular expression to find location
        location_reg = re.search(r'([^\d]+)?(\d{5})?', raw_text).group(0)
        
        # CB 7.16 - Come back and clean up. Don't have a good way to replace trailing punctuation yet
        location_reg = location_reg.replace('•','')

        # CB 7.17 - Replace name that is coming back in match and strip out white spaces
        location_reg = location_reg.replace(name,'').strip()

        # Return location if it ends with a digit 
        if location_reg[-1].isdigit():
            return location_reg
        
        # Else split on a new line and return the first part
        else:
            return location_reg.split('\n')[0]

    # CB 7.16 - Disable as we have exceeded API limit. Clean up code ot make it mine
    def get_skills(self, raw_text):
        stop_words = set(nltk.corpus.stopwords.words('english'))
        word_tokens = nltk.tokenize.word_tokenize(raw_text)
    
        # remove the stop words
        filtered_tokens = [w for w in word_tokens if w not in stop_words]
    
        # remove the punctuation
        filtered_tokens = [w for w in word_tokens if w.isalpha()]
    
        # generate bigrams and trigrams (such as artificial intelligence)
        bigrams_trigrams = list(map(' '.join, nltk.everygrams(filtered_tokens, 2, 3)))
        return bigrams_trigrams
    
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

    # CB 7.16 - Disable as this is not a clean solution. Clean up code to make it mine
    def get_education(self, raw_text):
        pass
        # RESERVED_WORDS = [
        #     'school',
        #     'college',
        #     'university',
        #     'academy',
        #     'faculty',
        #     'institute',
        #     'faculdades',
        #     'Schola',
        #     'schule',
        #     'lise',
        #     'lyceum',
        #     'lycee',
        #     'polytechnic',
        #     'kolej',
        #     'ünivers',
        #     'okul',
        # ]

        # organizations = []
 
        # # first get all the organization names using nltk
        # for sent in nltk.sent_tokenize(raw_text):
        #     for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
        #         if hasattr(chunk, 'label') and chunk.label() == 'ORGANIZATION':
        #             organizations.append(' '.join(c[0] for c in chunk.leaves()))
    
        # # we search for each bigram and trigram for reserved words
        # # (college, university etc...)
        # education = set()
        # for org in organizations:
        #     for word in RESERVED_WORDS:
        #         if org.lower().find(word):
        #             education.add(org)
    
        # return education


#Create class for text analysis
class AnalyzeText():
    # Initialize class
    def __init__(self):
        return
    
    # Tokenize, remove stop words and punctuation, lemmatize
    def tokenize_text(self, raw_text):

        # Create stop words and punctuation
        stop_words = set(nltk.corpus.stopwords.words('english'))
        punctuation = string.punctuation

        # CB 7.17 - Noticed some punctuation not remove so defining here
        user_defined_punctuation = ['–','•','’']

        # Create word_tokens from resume
        tokens = nltk.tokenize.word_tokenize(raw_text)
    
        # remove the stop words, string.punctuation, and user_defined_punctuation
        tokens = [w for w in tokens if w not in stop_words and w not in punctuation and w not in user_defined_punctuation]
    
        # Lemmatize text
        wordnet_lemmatizer = WordNetLemmatizer()
        tokens = [wordnet_lemmatizer.lemmatize(word).lower().strip() for word in tokens]

        # Return processed text joined together
        return " ".join([i for i in tokens])
    
    # Create features from processed tokens
    def word_count_features(self, processed_text):
        
        # Create empty list to store tfidf vectors depending on ngram_range
        tfidf_vector_list = []

        #Create empty list to store word counts from Bag-of-words
        count_vector_list =[]
        
        # Create df list to hold dataframes
        df_list = []

        # Conduct TFIDF Vectorization and Count Vectorization using different combinations of uniqrams, bigrams, and trigrams
        for i in range(1,4):
            tfidf_vec = TfidfVectorizer(ngram_range = (i,i))
            count_vec = CountVectorizer(ngram_range = (i,i))

            ### TFIDF ###
            # TFIDF - apply fit_transform with the processed text as an iterable to satisfy argument reqs
            tfidf_result = tfidf_vec.fit_transform([processed_text])

            # Store results to tfidf_vector_list if needed later
            tfidf_vector_list.append(tfidf_result) 

            # Convert from vector to matrix
            tfidf_matrix = tfidf_vec.fit_transform([processed_text]).todense()

            # Create feature index from matrix by getting words
            tfidf_feat_idx = tfidf_matrix[0,:].nonzero()[1]

            # Create zipped words to scores
            tfidf_scores = zip([tfidf_vec.get_feature_names()[i] for i in tfidf_feat_idx], [tfidf_matrix[0,x] for x in tfidf_feat_idx])

            # Create dictionary from scores
            tfidf_dict = dict(tfidf_scores)

            # Put into dataframe
            tfidf_df = pd.DataFrame(tfidf_dict.items(), columns = ['Phrase','Tf-idf Score'])

            ### Bag of words ###
            # Count - apply fit_transform with the processed text as an iterable to satisfy argument reqs
            count_result = count_vec.fit_transform([processed_text])

             # Store results to count_vector_list if needed later
            count_vector_list.append(count_result) 

            # Convert from vector to matrix
            count_matrix = count_vec.fit_transform([processed_text]).todense()

            # Create feature index from matrix by getting words
            count_feat_idx = count_matrix[0,:].nonzero()[1]

            # Create zipped words to scores
            count_scores = zip([count_vec.get_feature_names()[i] for i in count_feat_idx], [count_matrix[0,x] for x in count_feat_idx])

            # Create dictionary from scores
            count_dict = dict(count_scores)

            # Put into dataframe
            count_df = pd.DataFrame(count_dict.items(), columns = ['Phrase', 'Count'])

            # Join dfs on the phrase
            tfidf_df = tfidf_df.merge(count_df, left_on=['Phrase'], right_on =['Phrase'])

            # Append to df list
            df_list.append(tfidf_df)

        # Return list of dataframes for Tf-idf and Bag-of-words
        return df_list




        




    # Make cloud chunks from resume 
    # Takes in the raw text form resume and the extracted list of attributes
    def chunk_text(self, raw_text, attribute_dict):

        # Replace • placeholder
        
        return raw_text, attribute_dict