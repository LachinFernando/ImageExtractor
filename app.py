import streamlit as st
import retriever_utils as rutils


IMAGE_ADDRESS = "https://miro.medium.com/v2/resize:fit:1400/0*50gYIhjGFFd82BuF"

# web application
st.title("Image Extractor")

st.image(IMAGE_ADDRESS)

# set up the user query
user_query = st.text_input("Describe the Image you want in few words")

if user_query:
    # get the embeddings
    query_embedds = rutils.get_openai_embeddings(user_query)
    # fetch the closes record
    closest_records = rutils.query_response(query_embedds)
    # extract text and image
    closest_text, image = rutils.content_extractor(closest_records)

    # set the image
    st.subheader("Generated Image")
    st.image(image)
    # set up side bar
    with st.sidebar:
        st.subheader("Query Details")
        st.markdown("**User Query**: {}".format(user_query))
        st.markdown("**Closest Query**: {}".format(closest_text))
