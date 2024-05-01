import os
import time
from textwrap import dedent
import dash
from dash import html
from dash import dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from PIL import Image
import json
import re
import requests
from bs4 import BeautifulSoup
import plotly.express as px
import pandas as pd
# import nltk

import openai
hf_key = os.getenv("HF_API_KEY")
openai.api_key = "XXXXXXXXXX" ##INSERT YOUR OPENAI API KEY HERE


from primeqa.components.reranker.colbert_reranker import ColBERTReranker

model_name_or_path = "DrDecr.dnn"


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


bamToken = os.getenv('BAM_TOKEN')

def extract_desired_text(input_document, target_keyword):
    lines = input_document.split('\n')
    desired_text = ""
    last_occurrence_index = -1

    for i, line in enumerate(lines):
        if target_keyword in line:
            last_occurrence_index = i

    if last_occurrence_index != -1:
        desired_lines = lines[last_occurrence_index+1:]
        desired_text = "\n".join(line.strip() for line in desired_lines)
    else:
        desired_text = input_document.strip()

    return desired_text


def clean_text(text_data):
    replaced = re.sub("\{{ .*?\}}", "", text_data)
    replaced = re.sub("\{: .*?\}", "", replaced)
    replaced = re.sub("\.*?", "", replaced)
    replaced = re.sub("\(.*?\)|\[.*?\] |\{.*?\}", "", replaced)
    replaced = re.sub("</?div[^>]*>", "", replaced)
    replaced = re.sub("</?p[^>]*>", "", replaced)
    replaced = re.sub("</?a[^>]*>", "", replaced)
    replaced = re.sub("</?h*[^>]*>", "", replaced)
    replaced = re.sub("</?em*[^>]*>", "", replaced)
    replaced = re.sub("</?img*[^>]*>", "", replaced)
    replaced = re.sub("&amp;", "", replaced)
    replaced = re.sub("</?href*>", "", replaced)
    replaced = re.sub("\s+", " ", replaced)
    replaced = replaced.replace("}", "")
    replaced = replaced.replace("##", "")
    replaced = replaced.replace("###", "")
    replaced = replaced.replace("#", "")
    replaced = replaced.replace("*", "")
    replaced = replaced.replace("<strong>", "")
    replaced = replaced.replace("</strong>", "")
    replaced = replaced.replace("<ul>", "")
    replaced = replaced.replace("</ul>", "")
    replaced = replaced.replace("<li>", "")
    replaced = replaced.replace("</li>", "")
    replaced = replaced.replace("<ol>", "")
    replaced = replaced.replace("</ol>", "")
    return replaced



max_num_documents=10

def solr_retriever(question):
    solr_url = f'http://localhost:7574/solr/Manuals/select?q={question}&q.op=AND&wt=json'
    response = requests.get(solr_url)
    query_results = response.json()

    total_documents = query_results['response']['numFound']
    print(f"{total_documents} documents found.")

    results_list = []
    if total_documents > 0:
        total_documents = min(total_documents, 10)
        for i in range(total_documents):
            content_unicode = query_results['response']['docs'][i]['content'][0]
            content_decoded = content_unicode.encode("ascii", "ignore").decode()
            keyword = "{: shortdesc} "
            cleaned_text = extract_desired_text(content_decoded, keyword)
            pattern = r'\{\s*:\s*[\w#-]+\s*\}|\{\s*:\s*\w+\s*\}|\n\s*\n'
            cleaned_text = re.sub(pattern, '', cleaned_text)
            cleaned_text = clean_text(cleaned_text)

            document_info = {
                "document": {
                    "rank": i,
                    "document_id": query_results['response']['docs'][i]['id'][0],
                    "text": cleaned_text[1000:3000],
                },
            }
            results_list.append(document_info)

        results_to_display = [result['document'] for result in results_list]
        df = pd.DataFrame.from_records(results_to_display, columns=['rank', 'document_id', 'text'])
        df.dropna(inplace=True)

    print('======================================================================')
    print(f'QUERY: {question}')
    return results_list
   

def drdecr_reranker(question, max_reranked_documents=10):
    reranker = ColBERTReranker(model=model_name_or_path)
    reranker.load()

    search_results = solr_retriever(question)
    if len(search_results) > 0:
        reranked_results = reranker.predict(queries=[question], documents=[search_results], max_num_documents=max_reranked_documents)

        print(reranked_results)

        reranked_results_to_display = [result['document'] for result in reranked_results[0]]
        df = pd.DataFrame.from_records(reranked_results_to_display, columns=['rank', 'document_id', 'text'])
        print('======================================================================')
        print(f'QUERY: {question}')
        return df['text'][0]
    else:
        return "0 documents found", "None"


def Header(name, app):
    title = html.H1(name, style={"margin-top": 5})
    logo = html.Img(
        src=app.get_asset_url("apress-logo.png"), style={"float": "right", "height": 60}
    )
    return dbc.Row([dbc.Col(title, md=8), dbc.Col(logo, md=4)])


def textbox(text, box="AI", name="Knowa"):
    text = text.replace(f"{name}:", "").replace("You:", "")
    style = {
        "max-width": "60%",
        "width": "max-content",
        "padding": "5px 10px",
        "border-radius": 25,
        "margin-bottom": 20,
    }

    if box == "user":
        style["margin-left"] = "auto"
        style["margin-right"] = 0

        return dbc.Card(text, style=style, body=True, color="primary", inverse=True)

    elif box == "AI":
        style["margin-left"] = 0
        style["margin-right"] = "auto"

        thumbnail = html.Img(
            src=app.get_asset_url("apress-logo.png"),
            style={
                "border-radius": 50,
                "height": 36,
                "margin-right": 5,
                "float": "left",
            },
        )
        textbox = dbc.Card(text, style=style, body=True, color="light", inverse=False)

        return html.Div([thumbnail, textbox])

    else:
        raise ValueError("Incorrect option for `box`.")

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server


IMAGES = {"apress": app.get_asset_url("apress-logo.png")}


conversation = html.Div(
    html.Div(id="display-conversation"),
    style={
        "overflow-y": "auto",
        "display": "flex",
        "height": "calc(90vh - 132px)",
        "flex-direction": "column-reverse",
    },
)

controls = dbc.InputGroup(
    children=[
        dbc.Input(id="user-input", placeholder="Ask me anything about our products...", type="text", size="lg", className="mb-3"),
        dbc.Button("Submit", id="submit", className="mb-3"),
    ]
)

app.layout = dbc.Container(
    fluid=False,
    children=[
        Header("Generative Q&A", app),
        dbc.Row(html.H6("Enterprise Knowledge Bank", style={'textAlign': 'left'}),
                        className="me-auto",
                        align='left',
                        justify='let'
                    ),
        html.Hr(),
        dcc.Tabs(id="tabs-example-graph", value='tab-1-example-graph', children=[
        dcc.Tab(label='AI powered QA', children=[
                                            dcc.Store(id="store-conversation", data=""),
                                            conversation,
                                            controls,
                                            dbc.Spinner(html.Div(id="loading-component")),
                ]),
            ]),
        html.Br(),
        html.Footer(children="Please note that this content is made available with the Generative AI book. \
                        The content may include systems & methods pending patent with USPTO and protected under US Patent Laws. \
                        Copyright - Apress Publication")
    ],
)

@app.callback(
    Output("display-conversation", "children"),
    [Input("store-conversation", "data")]
)
def update_display(chat_history):
    conversation_parts = chat_history.split("<split>")[:-1]
    conversation_components = [
        textbox(x, box="user") if i % 2 == 0 else textbox(x, box="AI")
        for i, x in enumerate(conversation_parts)
    ]
    return conversation_components



@app.callback(
    Output("user-input", "value"),
    [Input("submit", "n_clicks"), Input("user-input", "n_submit")],
)
def clear_input(n_clicks, n_submit):
    return "" if n_clicks or n_submit else dash.no_update

def call_huggingface_api(model_input):
    API_URL = "https://api-inference.huggingface.co/models/bigscience/bloom"
    headers = {"Authorization": hf_key}

    json_data = {
        "inputs": model_input,
        "parameters": {'temperature': 0.5, 'max_new_tokens': 50, 'return_full_text': False},
    }

    response = requests.post(API_URL, headers=headers, json=json_data)
    json_response = json.loads(response.content.decode("utf-8"))
    model_output = json_response[0]['generated_text']

    return model_output

def call_openai_api(model_input):
    
    response = openai.Completion.create(
        engine="davinci",
        prompt=model_input,
        max_tokens=1000,
        temperature=0.9,
    )
    model_output = response.choices[0].text.strip()
    return model_output


def format_model_output(model_output):
    sentences = model_output.split(". ")
    unique_sentences = list(dict.fromkeys(sentences))

    if not model_output.endswith("."):
        unique_sentences.pop()

    model_output = ". ".join(unique_sentences) + "."
    
    return model_output

@app.callback(
    [Output("store-conversation", "data"), Output("loading-component", "children")],
    [Input("submit", "n_clicks"), Input("user-input", "n_submit")],
    [State("user-input", "value"), State("store-conversation", "data")],
)
def run_chatbot(n_clicks, n_submit, user_input, chat_history):
    if n_clicks == 0 and n_submit is None:
        return "", None

    if user_input is None or user_input == "":
        return chat_history, None
    
    context = drdecr_reranker(user_input)
    print("-----------context----------", context)
    chat_history += f"Answer the question based on the context below. " + \
        "Context: "  + context + \
        " Question: " + user_input

    model_input = chat_history.replace("<split>", "\n")
    
    model_output = call_huggingface_api(model_input)
    print("============model_output===============", model_output)
    model_output = model_output.replace("Answer the question based on the context", "")
    
    model_output = f"{model_output}<split>"
    
    return model_output, None

if __name__ == "__main__":
    app.run_server(port=8051, debug=True)