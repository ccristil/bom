import streamlit as st

st.title("About")


st.write("This app was inspired by a project for my Teachings and Doctrines of the Book of Mormon class at BYU."
         " Sometimes searching the scriptures can be hard, especially when we are in the midst of trials. I built this "
         "so that when someone is feeling down, all they have to do is enter their general emotion and then they can "
         "instantly have access to verses that relate to them. This isn't a perfect app so sometimes that model will give "
         "back verses that don't line up with the prompt but for the most part it gets the verses right. Enjoy the app! 😊"
         )

tools = ['python','SentenceTransformer','pandas','numpy','Hugging Face🤗']
st.header('Tools Used ⚒️')
st.write("During this project, I did a lot of research before anything was built and settled on trying out streamlit for both the UI"
         " as well as the hosting. This turned out to be a great investment of my time and I've loved building and deploying with streamlit.")
st.write("In addition to streamlit, here are the rest of the tools that I used to build this app:")
for tool in tools:
    st.write(f"•  {tool}")


st.header("How it works⚙️")

st.write("At a high level, when you input your prompt it gets sent a model that is built to recognize patterns and feeling"
         "in text. This particular model was trained on all of the standard works of the Church of Jesus Christ of Latter-Day "
         "Saints. The model will take your input and figure out what verses are most closely related to the 'feeling' that "
         "your input has. This means that it's not just matching keywords, but analyzing the deeper meaning of to get more holistic matches.")


st.write("For a more in depth look into how it works go [here](https://www.marqo.ai/course/introduction-to-sentence-transformers).")