import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from huggingface_hub import hf_hub_download
import torch
import random as rd
from helper_functions import get_model, url_abbreviations


placeholders = [
    "I feel lonely and I want some comfort...",
    "How can I find peace when I’m anxious?",
    "What do the scriptures say about forgiveness?",
    "I’m struggling with faith — what can help?",
    "How do I deal with anger or frustration?",
    "I want to feel closer to God — where do I start?",
    "What is the purpose of trials in life?",
    "How can I strengthen my testimony?",
    "What does the Bible teach about love?",
    "I feel lost — is there hope for me?"
]
if 'placeholder' not in st.session_state:
    st.session_state.placeholder = rd.choice(placeholders)
st.title("Home")

# initialize the embeddings and csv file
# embeddings_path = hf_hub_download(repo_id="ccristil/bom_embeddings", filename="bom_embeddings.npy",repo_type="dataset")
# df_path = hf_hub_download(repo_id="ccristil/bom_embeddings", filename="bom.csv", repo_type="dataset")
#
# embeddings = np.load(embeddings_path)
# df = pd.read_csv(df_path)

# device agnostic code
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# load data
model = SentenceTransformer('all-MiniLM-L6-v2')
# embeddings = np.load('data/bom_embeddings.npy')
# df = pd.read_csv('data/bom.csv')


### logic ###
def get_url(book_title, chapter_number, verse_number):


  volume_dict = {
      "Book of Mormon" : "bofm",
      "New Testament" : "nt",
      "Old Testament" : "ot",
      "Doctrine and Covenants" : "dc-testament/dc",
      "Pearl of Great Price" : "pgp"
  }
  volume = volume_dict[option]



  book = url_abbreviations[book_title]

  url = f"https://www.churchofjesuschrist.org/study/scriptures/{volume}/{book}/{chapter_number}?lang=eng&id=p{verse_number}#p{verse_number}"
  return url

def get_similar_verses(input, model, embeddings, num_to_return:int=5, print_result=False,option="Book of Mormon"):

  query_embedding = model.encode([input])

  similarities = cosine_similarity(query_embedding, embeddings)[0]

  top_indices = similarities.argsort()[-num_to_return:][::-1]

  if print_result:
    print(f'Printing the top {num_to_return} results...')
    for idx in top_indices:
      verse = df.iloc[idx]
      url = get_url(book_title=verse['book_title'],chapter_number=verse['chapter_number'],verse_number=verse['verse_number'])
      print(f"{verse['verse_title']}: {verse['scripture_text']} | URL: {url}")

  return top_indices


st.title("_✨ Ask and Ye Shall Find: Semantic Search for Scripture_")
st.write("Type in a question, emotion, or phrase — and discover relevant verses.")
volumes = ['Book of Mormon','New Testament','Old Testament','Doctrine and Covenants','Pearl of Great Price']


option = st.selectbox('📜 What volume of scripture would you like the verses to be from?',volumes)
# number = st.number_input("🔢 How many verses would you like?", 0, 10,5)

# if option == 'New Testament':
#     nt_embeddings_path = hf_hub_download(repo_id="ccristil/bom_embeddings", filename="nt_embeddings.npy", repo_type="dataset")
#     embeddings = np.load(nt_embeddings_path)
#
#     nt_df_path = hf_hub_download(repo_id="ccristil/bom_embeddings", filename="new_testament.csv", repo_type="dataset")
#     df = pd.read_csv(nt_df_path)



query = st.text_input("**🔍 What’s on your mind?**",
                      placeholder=st.session_state.placeholder)

df, embeddings = get_model(option=option)


if query:
    with st.spinner("Searching the scriptures..."):
        indices = get_similar_verses(input=query,
                                     model=model,
                                     embeddings=embeddings,
                                     option=option)
    for idx in indices:
        verse = df.iloc[idx]
        ref = verse['verse_title']
        text = verse['scripture_text']
        url = get_url(
            book_title=verse['book_title'],
            chapter_number=verse['chapter_number'],
            verse_number=verse['verse_number']
        )
        st.markdown(f"**[{ref}]({url})**")
        st.write(text)
        st.markdown("---")
