from huggingface_hub import hf_hub_download
from torch.utils.data.datapipes.gen_pyi import extract_method_name
import numpy as np
import pandas as pd

def get_model(option:str="Book of Mormon"):
    """
    This gets the model from the huggingface hub along with csv of
    the chosen volume of scripture.

    :param option: The volume of scripture you want.
    :return: (df, model)The model and csv of the scripture.
    """
    csv_dict = {"Book of Mormon" : "bom.csv",
                 "New Testament" : "new_testament.csv",
                 "Old Testament" : "old_testament.csv",
                 "Doctrine and Covenants" : "dc.csv",
                 "Pearl of Great Price" : "pearl_of_great_price.csv"}
    embedding_dict = {"Book of Mormon": "bom_embeddings.npy",
                "New Testament": "nt_embeddings.npy",
                "Old Testament": "ot_embeddings.npy",
                "Doctrine and Covenants": "dc_embeddings.npy",
                "Pearl of Great Price": "pgp_embeddings.npy"}

    csv_path = hf_hub_download(repo_id="ccristil/bom_embeddings", filename=csv_dict[option], repo_type="dataset")
    df = pd.read_csv(csv_path)

    embeddings_path = hf_hub_download(repo_id="ccristil/bom_embeddings", filename=embedding_dict[option], repo_type="dataset")
    embeddings = np.load(embeddings_path)


    return df, embeddings


# URL abbreviations used for building the proper URL to link to the verse.
url_abbreviations = {
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
    "Moroni": "moro",
    "Matthew": "matt",
    "Mark": "mark",
    "Luke": "luke",
    "John": "john",
    "Acts": "acts",
    "Romans": "rom",
    "1 Corinthians": "1-cor",
    "2 Corinthians": "2-cor",
    "Galatians": "gal",
    "Ephesians": "eph",
    "Philippians": "philip",
    "Colossians": "col",
    "1 Thessalonians": "1-thes",
    "2 Thessalonians": "2-thes",
    "1 Timothy": "1-tim",
    "2 Timothy": "2-tim",
    "Titus": "titus",
    "Philemon": "philem",
    "Hebrews": "heb",
    "James": "james",
    "1 Peter": "1-pet",
    "2 Peter": "2-pet",
    "1 John": "1-jn",
    "2 John": "2-jn",
    "3 John": "3-jn",
    "Jude": "jude",
    "Revelation": "rev",
    "Genesis": "gen",
    "Exodus": "ex",
    "Leviticus": "lev",
    "Numbers": "num",
    "Deuteronomy": "deut",
    "Joshua": "josh",
    "Judges": "judg",
    "Ruth": "ruth",
    "1 Samuel": "1-sam",
    "2 Samuel": "2-sam",
    "1 Kings": "1-kgs",
    "2 Kings": "2-kgs",
    "1 Chronicles": "1-chr",
    "2 Chronicles": "2-chr",
    "Ezra": "ezra",
    "Nehemiah": "neh",
    "Esther": "esth",
    "Job": "job",
    "Psalms": "ps",
    "Proverbs": "prov",
    "Ecclesiastes": "eccl",
    "Song of Solomon": "song",
    "Isaiah": "isa",
    "Jeremiah": "jer",
    "Lamentations": "lam",
    "Ezekiel": "ezek",
    "Daniel": "dan",
    "Hosea": "hosea",
    "Joel": "joel",
    "Amos": "amos",
    "Obadiah": "obad",
    "Jonah": "jonah",
    "Micah": "micah",
    "Nahum": "nahum",
    "Habakkuk": "hab",
    "Zephaniah": "zeph",
    "Haggai": "hag",
    "Zechariah": "zech",
    "Malachi": "mal",
    "Moses": "moses",
    "Abraham": "abr",
    "Joseph Smith--Matthew": "js-m",
    "Joseph Smith--History": "js-h",
    "Articles of Faith": "a-of-f",
    "Doctrine and Covenants": "dc",
    "Official Declaration 1": "od/1",
    "Official Declaration 2": "od/2"
  }