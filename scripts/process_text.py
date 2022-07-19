# Import functions
from scripts.api import skills_API

## Import libraries
import pandas as pd
import numpy as np
import re
import requests
import string
from datetime import datetime

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


#Create class for extracting text
class ExtractResumeText():
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
    # CB 7.17 - Add validation 
    def get_email(self, raw_text):
        
        # Use regular expression to find email
        #email_reg = re.compile(r'[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+')
        email_reg = re.compile(r'\S+@\S+')

        try:
            # Return first email found for the person
            return re.findall(email_reg, raw_text)[0]
        except:
            return 'Could not find email'

    # Extract phone number
    # CB 7.17 - Add validation 
    def get_phone(self, raw_text):
        
        # Use regular expression to find phone
        phone_reg = re.compile(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]')
        
        # Return first phone number found for the person
        return re.findall(phone_reg, raw_text)[0]

    # Get the candidate's location. CB 7.17 - This is very hacky, need to replace functionality for getting location
    # CB 7.17 - Add validation 
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
    
    # Create features from processed tokens (takes data, type of data being passed - default is resume, and name - default is blank)
    # CB 7.17 - Might have to strip out into 2 separate functions
    def build_word_count_features(self, data, data_type='resume', uploaded_file = 'Pete Fagan'):

        # Create df list to hold dataframes created during analysis
        df_list = []

        # If data being passed in for jobs, assign data as the processed_text column
        if data_type == 'jobs':

            # Create master list to hold all dfs
            master_df_list = []

            # Get the list of job titles
            title_list = list(data['Title'])

            # Assign data to the processed_text column that was already put through the pipelin
            data = list(data['processed_text'])
            
            # Loop through each tokenized post
            for text in data:
                # Loop through the range to get unigrams, bigrams, and trigrams
                for i in range(1,4):
                    
                    ### Initalize vector for tfidf and count
                    tfidf_vec = TfidfVectorizer(ngram_range = (i,i))
                    count_vec = CountVectorizer(ngram_range = (i,i))

                    ### Tf-idf ###
                    # Run fit_transform on the data (pass in as list)
                    tfidf_result = tfidf_vec.fit_transform([text])

                    # Map words to scores
                    tfidf_dict = dict(zip(tfidf_vec.get_feature_names(), tfidf_result.toarray()[0]))

                    # Put into dataframe
                    tfidf_df = pd.DataFrame(tfidf_dict.items(), columns = ['Phrase','Tf-idf Score'])

                    ### Bag of words ###
                    # Count - apply fit_transform to the data
                    count_result = count_vec.fit_transform([text])

                    # Map words to scores
                    count_dict = dict(zip(tfidf_vec.get_feature_names(), count_result.toarray()[0]))

                    # Put into dataframe
                    count_df = pd.DataFrame(count_dict.items(), columns = ['Phrase', 'Count'])

                    ### Join dfs on "Phrase" ###
                    tfidf_df = tfidf_df.merge(count_df, left_on=['Phrase'], right_on =['Phrase'])

                    # Append to df list
                    df_list.append(tfidf_df)
                
                # After processing a job post, append results to master_list
                master_df_list.append(df_list)
                # Clear df_list to prepare for next iteration
                df_list = []
            
            ### Write to Excel
            # Initialize title_counter
            title_counter = 0

            # Loop through each post in the master_df_list
            for post_list in master_df_list:

                # Initialize appended_df for writing to excel
                appended_df = pd.DataFrame()

                # Loop through each dataframe for the post. Need to iterate through a range for modulo
                for i in range(len(post_list)):
                    df = post_list[i]

                    # If first dataframe for the post, set appended_df to the unigram dataframe
                    if i % 3 == 0:
                        # Create type column for analysis and set to Unigram
                        df['Type'] = 'Unigram'
                        appended_df = df
                        
                    # If second dataframe for the post, append onto appended_df
                    elif i % 3 == 1:
                        # Create type column for analysis and set to Bigram
                        df['Type'] = 'Bigram'

                        # Concatenate dataframes
                        appended_df = pd.concat([appended_df, df])

                    elif i % 3 == 2:
                        # Create type column for analysis and set to Trigram
                        df['Type'] = 'Trigram'

                        # Concatenate dataframes
                        appended_df = pd.concat([appended_df, df])
                    
                        # Sort appended df before writing to Excel 
                        appended_df = appended_df.sort_values(by=['Tf-idf Score'], ascending = False)

                        # Write the appended df for the job post to Excel
                        # CB 7.17 - Might need some sort of ID to link the postings to the features in case there are duplicates
                        # CB 7.17 - Would involve update the other write to excel code in employer_postings.py

                        # Need to replace slashes with "_" to save features. Remember this when reading and matching titles
                        appended_df.to_excel(
                            'data//postings//archive/archived_features//{}_{}.xlsx'.format(
                                title_list[title_counter].replace('/','_'),
                                datetime.now().strftime("%Y-%m-%d")),
                                index=False
                            )

                         # Need to replace slashes with "_" to save features. Remember this when reading and matching titles
                        appended_df.to_excel(
                            'data//postings//features//{}.xlsx'.format(title_list[title_counter].replace('/','_')),
                            index=False
                            )
                
                # Iterate title counter for next job post
                title_counter = title_counter + 1

            # Empty return
            return


        # Else, process resumes
        else: 

            # Conduct TFIDF Vectorization and Count Vectorization using different combinations of uniqrams, bigrams, and trigrams
            for i in range(1,4):
                tfidf_vec = TfidfVectorizer(ngram_range = (i,i))
                count_vec = CountVectorizer(ngram_range = (i,i))

                ### TFIDF ###
                # TFIDF - apply fit_transform to the data
                tfidf_result = tfidf_vec.fit_transform(data)

                # Map words to scores
                tfidf_dict = dict(zip(tfidf_vec.get_feature_names(), tfidf_result.toarray()[0]))

                # # Put into dataframe
                tfidf_df = pd.DataFrame(tfidf_dict.items(), columns = ['Phrase','Tf-idf Score'])

                ### Bag of words ###
                # Count - apply fit_transform to the data
                count_result = count_vec.fit_transform(data)

                # Map words to scores
                count_dict = dict(zip(tfidf_vec.get_feature_names(), count_result.toarray()[0]))

                # Put into dataframe
                count_df = pd.DataFrame(count_dict.items(), columns = ['Phrase', 'Count'])

                ### Join dfs on "Phrase" ###
                tfidf_df = tfidf_df.merge(count_df, left_on=['Phrase'], right_on =['Phrase'])

                # Append to df list
                df_list.append(tfidf_df)

            ### Write to Excel
            # Initialize appended_df for writing to excel
            appended_df = pd.DataFrame()

            # Loop through each dataframe for the post. Need to iterate through a range for modulo
            for i in range(len(df_list)):
                df = df_list[i]

                # If first dataframe for the post, set appended_df to the unigram dataframe
                if i % 3 == 0:
                    # Create type column for analysis and set to Unigram
                    df['Type'] = 'Unigram'
                    appended_df = df
                    
                # If second dataframe for the post, append onto appended_df
                elif i % 3 == 1:
                    # Create type column for analysis and set to Bigram
                    df['Type'] = 'Bigram'

                    # Concatenate dataframes
                    appended_df = pd.concat([appended_df, df])

                elif i % 3 == 2:
                    # Create type column for analysis and set to Trigram
                    df['Type'] = 'Trigram'

                    # Concatenate dataframes
                    appended_df = pd.concat([appended_df, df])
                
                    # Sort appended df before writing to Excel 
                    appended_df = appended_df.sort_values(by=['Tf-idf Score'], ascending = False)

                    # Write the appended df for the resume to Excel
                    # Need to replace slashes with "_" to save features. Remember this when reading and matching titles
                    appended_df.to_excel(
                        'data//resumes//features//{}.xlsx'.format(
                            uploaded_file.name.replace('/','_')),
                            index=False
                        )
                        
            # Empty Return
            return 

    # Make cloud chunks from resume 
    # Takes in the raw text form resume and the extracted list of attributes
    def chunk_text(self, raw_text, attribute_dict):

        # Replace • placeholder
        
        return raw_text, attribute_dict