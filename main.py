# import pandas as pd
# import numpy as np
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# from huggingface_hub import hf_hub_download
# import torch
import streamlit as st

# initialize the navbar to include all of the pages
pages = {
    "Navigation" : [
        st.Page("home.py"),
        st.Page("about.py"),
        st.Page("contact.py")
    ]
 }
pg = st.navigation(pages)
pg.run()
