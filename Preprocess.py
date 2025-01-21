import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

class Preprocess:
    def __init__(self, csv_file, kcs_file):
        self.csv_file = csv_file
        self.kcs_file = kcs_file
        

    def preprocess_data(self):
        # Read the CSV file
        cases_df = pd.read_csv(self.csv_file)
        kcs_df = pd.read_csv(self.kcs_file, encoding='ISO-8859-1', dtype={'Article Number': str, 'Case Number': str})
        #jira_df = pd.read_csv(self.jira_file)
         # Print column names for debugging
        print("Columns in CSV for cases:", cases_df.columns)
        print("Columns in CSV for kcs:", kcs_df.columns)

            # Fill NaN values with an empty string
        kcs_df['Title'] = kcs_df['Title'].fillna('')
        kcs_df['Description'] = kcs_df['Description'].fillna('')

        combined_text = kcs_df['Title'] + " " + kcs_df['Description']

        # Vectorize the combined text field for every enquiry
        general_vectorizer = TfidfVectorizer(stop_words='english')
        #general_vectors = general_vectorizer.fit_transform(df['combined'])
        general_vectors = general_vectorizer.fit_transform(cases_df['Problem/Issue'])

        # Vectorize the case numbers for case number-specific search
        case_number_vectorizer = TfidfVectorizer()
        case_number_vectors = case_number_vectorizer.fit_transform(cases_df['Case Number'].astype(str))

        # Vectorize KCS Articles
        kcs_vectorizer = TfidfVectorizer(stop_words='english')
        kcs_vectors = kcs_vectorizer.fit_transform(combined_text)

        # Vectorize Jira Cases
        #jira_vectorizer = TfidfVectorizer(stop_words='english')
        #jira_vectors = jira_vectorizer.fit_transform(jira_df['Problem'])

        # Save the vectorizer and vectors to disk
        with open('general_vectorizer.pkl', 'wb') as f:
            pickle.dump(general_vectorizer, f)
        with open('general_vectors.pkl', 'wb') as f:
            pickle.dump(general_vectors, f)
        
        with open('case_number_vectorizer.pkl', 'wb') as f:
             pickle.dump(case_number_vectorizer, f)
        with open('case_number_vectors.pkl', 'wb') as f:
             pickle.dump(case_number_vectors, f)
        
        with open('kcs_vectorizer.pkl', 'wb') as f:
            pickle.dump(kcs_vectorizer, f)
        with open('kcs_vectors.pkl', 'wb') as f:
            pickle.dump(kcs_vectors, f)

       # with open('jira_vectorizer.pkl', 'wb') as f:
       #     pickle.dump(jira_vectorizer, f)
       # with open('jira_vectors.pkl', 'wb') as f:
       #     pickle.dump(jira_vectors, f)

        return cases_df, kcs_df
