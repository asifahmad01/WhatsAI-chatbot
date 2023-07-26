import os
#import uvicorn
import requests

from flask import Flask, request
from fastapi import FastAPI

from flask import Flask, render_template, request, jsonify
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain import OpenAI
from langchain.chat_models import ChatOpenAI

from dotenv import load_dotenv
from pathlib import Path





from langchain import OpenAI
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter


load_dotenv(Path("./.env"))

# Set API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")





app = Flask(__name__)

# sending request
token = os.getenv('TOKEN')

# verify webhook
mytoken = os.getenv('MYTOKEN')








def chat(query):

    llm_gpt = ChatOpenAI(
    model_name='gpt-3.5-turbo',
    max_tokens=500
)
    



    vectordb = Chroma(persist_directory = 'vectorStore', collection_name = "my_collection",embedding_function = OpenAIEmbeddings())
    retriever = vectordb.as_retriever()
    #llm=OpenAI(temperature=0.1)
    qa = RetrievalQA.from_chain_type(llm=llm_gpt,chain_type="stuff",retriever = retriever)
    response = qa.run(query)
    return response



@app.route('/train', methods=['GET'])
def train():
    loader = TextLoader('context.txt')
    embeddings = OpenAIEmbeddings()

    index = VectorstoreIndexCreator(
        # split the documents into chunks
        text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=200,),
        # select which embeddings we want to use
        embedding=embeddings,
        # use Chroma as the vectorestore to index and search embeddings
        vectorstore_cls=Chroma,
        vectorstore_kwargs={"persist_directory": "vectorStore", "collection_name":"my_collection"}
    ).from_loaders([loader])
    return jsonify({'status': 'data loaded'})











@app.route("/webhook", methods=["GET"])
def verify_webhook():
    mode = request.args.get("hub.mode")
    challenge = request.args.get("hub.challenge")
    verify_token = request.args.get("hub.verify_token")

    if mode and verify_token:
        if mode == "subscribe" and verify_token == mytoken:
            return challenge, 200
        else:
            return "", 403




@app.route("/webhook", methods=["POST"])
def process_webhook():
    body_param = request.get_json()

    if body_param.get("object"):
        if (
            body_param.get("entry")
            and body_param["entry"][0].get("changes")
            and body_param["entry"][0]["changes"][0].get("value")
            and body_param["entry"][0]["changes"][0]["value"].get("messages")
            and body_param["entry"][0]["changes"][0]["value"]["messages"][0]
        ):
            phone_no_id = body_param["entry"][0]["changes"][0]["value"]["metadata"]["phone_number_id"]
            sender = body_param["entry"][0]["changes"][0]["value"]["messages"][0]["from"]
            message_body = body_param["entry"][0]["changes"][0]["value"]["messages"][0]["text"]["body"]

            print(chat(message_body))
            print(chat("hello"))
            print(message_body)
            
            response = requests.post(
                f"https://graph.facebook.com/v17.0/{phone_no_id}/messages?access_token={token}",
                json={
                    "messaging_product": "whatsapp",
                    "to": sender,
                    "text": {
                        "body": chat(message_body)
                    }
                },
                headers={
                    "Content-Type": "application/json"
                }
            )

            if response.status_code == 200:
                return "", 200
            else:
                return "", 500
        else:
            return "", 404
    else:
        return "", 404
        
@app.route("/", methods=["GET"])
def hello():
    return "hello there It's working fine", 200



@app.route("/testchat", methods=["GET"])
def test_chat_route():
    return chat("What are the names of five good friends in Jaipur?"), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0')