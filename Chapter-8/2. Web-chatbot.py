import streamlit as st
import requests
import re
from bs4 import BeautifulSoup
import json

QA_ENDPOINT = "https://api-inference.huggingface.co/models/bigscience/bloom"

SUMMARIZATION_ENDPOINT = "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct"

min_length = 100
max_length = 2000

def extract_combined_passage(url, min_paragraph_length, max_combined_length):
    response = requests.get(url)
    html_content = response.text

    soup = BeautifulSoup(html_content, "html.parser")

    paragraphs = soup.find_all("p")

    passages = [p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) >= min_paragraph_length]

    combined_passage = ""
    for passage in passages:
        if len(combined_passage) + len(passage) <= max_combined_length:
            combined_passage += passage + " "
        else:
            break

    return combined_passage

def main():
    st.title("Website Summarizer and Q&A")
    
    url = st.text_input("Enter the URL of a website:")
    question = st.text_input("Ask a question based on the text:")
    
    if st.button("Summarize and Answer"):
        if url:
            text = extract_combined_passage(url, min_length, max_length)
            
            summary = generate_summary(url)
            
            answer = answer_question(url, question)
            
            st.subheader("Summary:")
            st.write(summary)
            
            st.subheader("Answer:")
            st.write(answer)
        else:
            st.warning("Please enter a valid URL.")

def generate_summary(url):
    headers = {"Authorization": "Bearer XXXXXXXXXXXXX"} # INSERT YOUR HUGGINGFACE API KEY HERE
    combined_passage = extract_combined_passage(url, min_length, max_length)

    model_input = f"Summarize the following article. " + \
        "Article: "  + combined_passage

    json_data = {
            "inputs": model_input,
            "parameters": {'temperature': 0.5,
                        'max_new_tokens': 100,
                        'return_full_text': False,
                        },
        }
    
    response = requests.post(SUMMARIZATION_ENDPOINT, headers=headers, json=json_data)
    json_response = json.loads(response.content.decode("utf-8"))
    summary = json_response[0]['generated_text']
    return summary

def answer_question(url, question):
    headers = {"Authorization": "Bearer XXXXXXXXXXXXXXX"} # INSERT YOUR HUGGINGFACE API KEY HERE

    combined_passage = extract_combined_passage(url, min_length, max_length)

    model_input = f"Answer the question based on the context below. " + \
        "Context: "  + combined_passage + \
        " Question: " + question

    json_data = {
            "inputs": model_input,
            "parameters": {'temperature': 0.5,
                        'max_new_tokens': 100,
                        'return_full_text': False,
                        },
        }

    response = requests.post(QA_ENDPOINT, headers=headers, json=json_data)
    print("===================", response.json())

    answer = response.json()[0]["generated_text"]
    return answer

def scrape_text_from_website(url):

    r = requests.get(url)
    
    soup = BeautifulSoup(r.content, 'html.parser')
    s = soup.find('div', id= 'guide_contents')

    context_doc = ''
    for ele in s:
        context_doc += ele.text.strip()
    print(context_doc)

    clean_context = re.sub(r'[^a-zA-Z0-9.]', ' ', context_doc)
    print("+++++++++++SCRAPING OUTPUT++++++++++", clean_context)
    return clean_context

if __name__ == "__main__":
    main()
