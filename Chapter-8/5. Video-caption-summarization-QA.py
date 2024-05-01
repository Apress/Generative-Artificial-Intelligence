import streamlit as st
import requests
import urllib.parse
from langchain.document_loaders import YoutubeLoader
import json 
import re

SUMMARIZATION_ENDPOINT = "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct"
QA_ENDPOINT = "https://api-inference.huggingface.co/models/gpt2-large"

def main():
    st.title("YouTube Caption Summarizer and Q&A")

    url = st.text_input("Enter the YouTube URL:")
    question = st.text_input("Ask a question based on the caption:")
    
    if st.button("Summarize and Answer"):
        if url:
            video_id = extract_video_id(url)
            print("VIDEO_ID", video_id)
            

            transcript = get_youtube_captions(video_id)
            print("======TRANSCRIPT", transcript)
            
            answer = answer_question(transcript, question)
            print("**********", answer)

            summary = generate_summary(transcript)
            print("---------SUMMARY", summary)
            
            st.subheader("Summary:")
            st.write(summary)
            
            st.subheader("Answer:")
            st.write(answer)
        else:
            st.warning("Please enter a valid YouTube URL.")

def extract_video_id(url):

    # url = "https://www.youtube.com/watch?v=bZQun8Y4L2A&ab_channel=MicrosoftDeveloper"

    parsed_url = urllib.parse.urlparse(url)
    query_string = urllib.parse.parse_qs(parsed_url.query)

    video_id = query_string["v"][0]

    return video_id


def get_youtube_captions(video_id):
    loader = YoutubeLoader(video_id, language="en")
    summarization_docs = loader.load_and_split()
    summarization_text = summarization_docs[0].page_content
    summarization_text = summarization_text[0:2000]

    cleaned_text = re.sub(r"\[.*?\]", "", summarization_text)  # Remove square brackets and their contents
    cleaned_text = re.sub(r"\(.*?\)", "", cleaned_text)  # Remove parentheses and their contents
    cleaned_text = re.sub(r"\'s", "", cleaned_text)  # Remove apostrophe followed by 's'
    cleaned_text = re.sub(r"\w+://\S+", "", cleaned_text)  # Remove URLs

    unwanted_chars = ["'", '"', ",", ".", "!", "?"]
    cleaned_text = ''.join(c for c in cleaned_text if c not in unwanted_chars)

    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
    
    return cleaned_text

def generate_summary(context):
    headers = {"Authorization": "Bearer XXXXXXXXXXXXXXX"} ## INSERT YOUR HUGGGINGFACE API KEY HERE

    model_input = f"Summarize the following transcript. " + \
        "Transcript: "  + context

    json_data = {
            "inputs": model_input,
            "parameters": {'temperature': 0.5,
                        'max_new_tokens': 100,
                        'return_full_text': False,
                        },
        }

    response = requests.post(SUMMARIZATION_ENDPOINT, headers=headers, json=json_data)
    json_response = json.loads(response.content.decode("utf-8"))
    print("-----------------------------------", json_response)
    summary = json_response[0]['generated_text']
    return summary

def answer_question(context, question):
    headers = {"Authorization": "Bearer XXXXXXXXXXXXXXX"} ## INSERT YOUR HUGGGINGFACE API KEY HERE

    model_input = f"Answer the question based on the context below. " + \
        "Context: "  + context + \
        " Question: " + question

    json_data = {
            "inputs": model_input,
            "parameters": {'temperature': 0.5,
                        'max_new_tokens': 100,
                        'return_full_text': False,
                        },
        }

    response = requests.post(QA_ENDPOINT, headers=headers, json=json_data)

    answer = response.json()[0]["generated_text"]
    return answer

if __name__ == "__main__":
    main()