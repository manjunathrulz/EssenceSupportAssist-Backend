import os
from flask import Flask, request, jsonify, render_template
import pandas as pd
import json
import re
import nltk
from azure.storage.blob import BlobServiceClient
from nltk.corpus import stopwords
from Preprocess import Preprocess
from ProcessSimilaritySearch import SimilaritySearch

ProcessMain = Flask(__name__)

#nltk.download('stopwords')
#nltk.download('punkt_tab')

nltk.data.path.append(os.path.join(os.path.dirname(__file__), 'nltk_data'))


# Load the CSV file once
#csv_file = 'C:/Hey/GenAi/Documents/output3.csv'
#df = pd.read_csv(csv_file)

connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')

# Initialize BlobServiceClient
blob_service_client = BlobServiceClient.from_connection_string(connection_string)
container_client = blob_service_client.get_container_client("essencecs-genai-data")
blob_client_cases = container_client.get_blob_client("output3.csv")
blob_client_kcs = container_client.get_blob_client("KCS_Articles.csv")

# Download the CSV file
with open("output3.csv", "wb") as download_file:
    download_file.write(blob_client_cases.download_blob().readall())

df_cases = pd.read_csv("output3.csv")

#df_cases = pd.read_csv("C:/Hey/GenAi/Documents/output3.csv")

# Download the CSV file
with open("KCS_Articles.csv", "wb") as download_file:
    download_file.write(blob_client_kcs.download_blob().readall())


df_kcs_articles = pd.read_csv('KCS_Articles.csv', encoding='ISO-8859-1', dtype={'Article Number': str, 'Case Number': str})
#df_jira_cases = pd.read_csv('Jira_Cases.csv')

# Initialize preprocessing activity instead of every execution
#preprocess = Preprocess('output3.csv','KCS_Articles.csv','Jira_Cases.csv')
preprocess = Preprocess('output3.csv','KCS_Articles.csv')

cases_df, kcs_df = preprocess.preprocess_data()

@ProcessMain.route('/')
def index():
    return render_template('index2.html')

@ProcessMain.route('/search', methods=['POST'])
def search():
    print("Received request at /search")
    data = request.json
    print("Request data:", data)

    query = data['query']
    print("before stop words : ", query)
    cleanquery = clean_text(query)
    print(cleanquery)
    response_json = similarity_search(cleanquery)
    
    print("Response JSON:", response_json)
    return response_json

def clean_text(input_text):
    STOPWORDS = set(stopwords.words('english'))
    STOPWORDS_CUSTOM = set(['hello', 'hi', 'please', 'help', 'case'])
    REPLACE_BY_SPACE_RE = re.compile('[/(){}\\[\\]\\[@,;]')
    BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')

    # Remove symbols and other unrelated characters
    input_text = input_text.lower()
    input_text = REPLACE_BY_SPACE_RE.sub(' ', input_text)
    input_text = BAD_SYMBOLS_RE.sub('', input_text)
    input_text = ' '.join(word for word in input_text.split() if word not in STOPWORDS)
    cleansed_text = ' '.join(word for word in input_text.split() if word not in STOPWORDS_CUSTOM)

    return cleansed_text

def similarity_search(cleanquery):
    
    search = SimilaritySearch(cleanquery)
    top_kcs_results = []
    top_cases_results = []
    # Check if cleanquery contains a 7-digit number starting with 1 or 2
    if re.match(r'^[12]\d{6}$', cleanquery):
        print("Regular Expression matched")
        # Perform similarity search on case number
        top_cases_results = search.search_case_number()
        summaries_cases = prepare_cases_response(top_cases_results)
        summaries_kcs = {
            'Article Number': '',
            'Title': '',
            'Workaround/Fix': '',
            'Similarity Score': ''
        }
        print("Summery of case if block", summaries_cases)
        response = {
            'response_cases': summaries_cases,
            'response_kcs': summaries_kcs
         }
        response_json = json.dumps(summaries_cases)
        return response_json
    else:
        print("General Enquiry")
        # Perform the existing similarity search
        top_cases_results = search.search_cases()
        top_kcs_results = search.search_kcs()
        summaries_cases = prepare_cases_response(top_cases_results)
        summaries_kcs = prepare_kcs_response(top_kcs_results)
        response = {
            'response_cases': summaries_cases,
            'response_kcs': summaries_kcs
         }
        response_json = json.dumps(response)  
        return response_json

def prepare_cases_response(top_cases_results):

     # Retrieve the full rows
    top_indices = [result[0] for result in top_cases_results]
    results = cases_df.iloc[top_indices]

    # Summarize the full text
    summaries_cases = []
    for result, row in zip(top_cases_results, results.iterrows()):
        index, score = result
        _, row_data = row
        data = {
            'Case Number': row_data['Case Number'],
            'Issue': row_data['Problem/Issue'],
            'Version': row_data['Reported Version'],
            'Steps to Reproduce': row_data['Steps to Reproduce'],
            'Solution': row_data['Solution'],
            'ViewerURL': row_data['ViewerURL'],
            'Similarity Score': score
        }
        summaries_cases.append(data) 
        print("Added summary For Cases:", data) 

    print("Summaries (Cases):", summaries_cases)
    return summaries_cases

def prepare_kcs_response(top_kcs_results):
     # Retrieve the full rows for KCS
    top_indices_kcs = [result[0] for result in top_kcs_results]
    results_kcs = kcs_df.iloc[top_indices_kcs]

    print("KCS details", results_kcs)
        # Summarize the full text for KCS
   # Summarize the full text for KCS
    summaries_kcs = []  
    for result, row in zip(top_kcs_results, results_kcs.iterrows()):
        index, score = result
        _, row_data = row
        kcs_data = {
            'Article Number': row_data['Article Number'],
            'Title': row_data['Title'],
            'Workaround/Fix': row_data['Workaround/Fix'],
            'Similarity Score': score
        }
        summaries_kcs.append(kcs_data) 
        print("Added summary for KCS:", kcs_data) 

    print("Summaries (KCS):", summaries_kcs)

    return summaries_kcs


if __name__ == '__main__':
    print("Starting Flask app...") 
    #ProcessMain.run(debug=True)
    ProcessMain.run(host='0.0.0.0', port=5000)
    