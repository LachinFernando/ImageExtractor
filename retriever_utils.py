import os
import uuid

from openai import OpenAI
from pinecone import Pinecone
import streamlit as st


TEXT_MODEL = "text-embedding-ada-002"
NAMESPACE_KEY = "Sanskriti"

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["PINECONE_API_KEY"]= st.secrets["PINECONE_API_KEY"]
os.environ["INDEX_HOST"]= st.secrets["INDEX_HOST"]


pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
pc_index = pc.Index(host=os.environ["INDEX_HOST"])

# create client
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def get_openai_embeddings(text: str) -> list[float]:
    response = client.embeddings.create(input=f"{text}", model=TEXT_MODEL)

    return response.data[0].embedding


# function query similar chunks
def query_response(query_embedding, k = 1, namespace_ = NAMESPACE_KEY):
    query_response = pc_index.query(
        namespace=namespace_,
        vector=query_embedding,
        top_k=k,
        include_values=False,
        include_metadata=True,
    )

    return query_response


def content_extractor(similar_data: dict) -> tuple:
    top_value = similar_data["matches"][0]
    # get the text out
    text_content = top_value["metadata"]["text"]
    image_link = top_value["metadata"]["image_url"]

    return (text_content, image_link)