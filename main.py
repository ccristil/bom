# import pandas as pd
# import numpy as np
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# from huggingface_hub import hf_hub_download
# import torch
import streamlit as st

st.set_page_config(
    page_title = "BoM App",
    page_icon = "images/bom_logo.png",
    layout = "wide",
)
st.logo('images/bom_logo.png')
# initialize the navbar to include all of the pages
pages = [

        st.Page("pages/home.py", icon=":material/home:"),
        st.Page("pages/about.py", icon=":material/menu_book:"),
        st.Page("pages/contact.py", icon=":material/contact_page:")

 ]
pg = st.navigation(pages)
pg.run()
