# Import funcitons
from scripts.api import skills_API

## Import libraries
import pandas as pd
import re
import requests

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
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('vader_lexicon')


#Create class for text pipeline
class TextPipeline():
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
    
    # CB 7.16 - Extract name - need to fix intelligence
    def get_name(self, raw_text):

        for sent in nltk.sent_tokenize(raw_text):
            for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
                if hasattr(chunk, 'label') and chunk.label() == 'PERSON':
                    person_name =' '.join(chunk_leave[0] for chunk_leave in chunk.leaves())
        
        # Placeholder to just get the first line from the resume and assume it's the first name
        return raw_text.split('\n')[0]

    # Extract email
    def get_email(self, raw_text):
        
        # Use regular expression to find email
        EMAIL_REG = re.compile(r'[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+')

        # Return first email found for the person
        return re.findall(EMAIL_REG, raw_text)[0]

    # Extract phone number
    def get_phone(self, raw_text):
        
        # Use regular expression to find phone
        PHONE_REG = re.compile(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]')
        
        # Return first phone number found for the person
        return re.findall(PHONE_REG, raw_text)[0]

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
        RESERVED_WORDS = [
            'school',
            'college',
            'university',
            'academy',
            'faculty',
            'institute',
            'faculdades',
            'Schola',
            'schule',
            'lise',
            'lyceum',
            'lycee',
            'polytechnic',
            'kolej',
            'ünivers',
            'okul',
        ]

        organizations = []
 
        # first get all the organization names using nltk
        for sent in nltk.sent_tokenize(raw_text):
            for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
                if hasattr(chunk, 'label') and chunk.label() == 'ORGANIZATION':
                    organizations.append(' '.join(c[0] for c in chunk.leaves()))
    
        # we search for each bigram and trigram for reserved words
        # (college, university etc...)
        education = set()
        for org in organizations:
            for word in RESERVED_WORDS:
                if org.lower().find(word):
                    education.add(org)
    
        return education
