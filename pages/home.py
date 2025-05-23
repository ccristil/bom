import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from huggingface_hub import hf_hub_download
import torch

st.title("Home")

# initialize the embeddings and csv file
embeddings_path = hf_hub_download(repo_id="ccristil/bom_embeddings", filename="bom_embeddings.npy",repo_type="dataset")
df_path = hf_hub_download(repo_id="ccristil/bom_embeddings", filename="bom.csv", repo_type="dataset")

embeddings = np.load(embeddings_path)
df = pd.read_csv(df_path)

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
  bom_url_abbreviations = {
    "1 Nephi": "1-ne",
    "2 Nephi": "2-ne",
    "Jacob": "jacob",
    "Enos": "enos",
    "Jarom": "jarom",
    "Omni": "omni",
    "Words of Mormon": "w-of-m",
    "Mosiah": "mosiah",
    "Alma": "alma",
    "Helaman": "hel",
    "3 Nephi": "3-ne",
    "4 Nephi": "4-ne",
    "Mormon": "morm",
    "Ether": "ether",
    "Moroni": "moro"
  }

  book = bom_url_abbreviations[book_title]

  url = f"https://www.churchofjesuschrist.org/study/scriptures/bofm/{book}/{chapter_number}?lang=eng&id=p{verse_number}#p{verse_number}"
  return url

def get_similar_verses(input, model, embeddings, num_to_return:int=5, print_result=False):

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

# options = st.selectbox('📜 What volume of scripture would you like the verses to be from?',['Book of Mormon','New Testament', 'Old Testament', 'Doctrine & Covenants', 'Pearl of Great Price'])
# number = st.number_input("🔢 How many verses would you like?", 0, 10,5)


query = st.text_input("**🔍 What’s on your mind?**")

if query:
    with st.spinner("Searching the scriptures..."):
        indices = get_similar_verses(input=query, model=model, embeddings=embeddings)
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
