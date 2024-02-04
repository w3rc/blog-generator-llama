import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers


def getLlamaResponse(input_text, no_words, blog_style):
  llm = CTransformers(model="models/llama-2-7b-chat.ggmlv3.q8_0.bin", model_type="llama", config={'max_new_tokens': 256, 'temperature': 0.01})
  
  template = """
    Write a blog for {blog_style} job profile on {input_text} within {no_words} words.
  """
  
  prompt = PromptTemplate(input_variables=["blog_style", "input_text", "no_words"], template=template)
  response = llm.invoke(prompt.format(blog_style=blog_style, input_text=input_text, no_words=no_words))
  return response



st.set_page_config(layout="centered", page_title="Generate Blogs", page_icon="🤖", initial_sidebar_state="collapsed")
st.header("Generate Blogs 🤖")

input_text = st.text_area("Enter the text")

col1, col2 = st.columns([5,5])

with col1:
  no_words = st.number_input("Number of words", min_value=50, max_value=500, value=100)
with col2:
  blog_style = st.selectbox("Blog Style", ["Research", "Data Scientist", "Casual"], index=0)
  
submit = st.button("Generate")

if submit:
  st.write(getLlamaResponse(input_text, no_words, blog_style))