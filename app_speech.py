#------------------------------------------

import streamlit as st
import speech_recognition as sr
import os
import numpy as np
import json

with open("config.json") as f:
    config = json.load(f)

from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

#------------------------------------------

os.environ["OPENAI_API_KEY"] = config['OPENAI_API_KEY']

# Download embeddings from OpenAI
embeddings = OpenAIEmbeddings()

#------------------------------------------

def transcribir_audio(archivo_audio):
    recognizer = sr.Recognizer()
    with sr.AudioFile(archivo_audio) as source:
        audio = recognizer.record(source)  # Leer el archivo de audio
    try:
        texto = recognizer.recognize_google(audio, language="es-ES")  # Reconocer el texto en español
        return texto
    except sr.UnknownValueError:
        return "No se pudo entender el audio"
    except sr.RequestError as e:
        return f"No se pudo obtener respuesta del servicio de reconocimiento de voz; {e}"

def resumen_audio(texto):
    try:
        text_splitter = CharacterTextSplitter(
            separator = "\n",
            chunk_size = 1000,
            chunk_overlap  = 200,
            length_function = len,
        )
        texts = text_splitter.split_text(texto)
        docsearch = FAISS.from_texts(texts, embeddings)
        chain = load_qa_chain(OpenAI(), chain_type="stuff")

        query = "Lista las ideas más importantes?"
        docs = docsearch.similarity_search(query)
        respuesta=chain.run(input_documents=docs, question=query)

        return respuesta
    except sr.UnknownValueError:
        return "No se pudo resumir el audio"
    except sr.RequestError as e:
        return f"No se pudo obtener respuesta del servicio de resumen de texto; {e}"

# Configuración de la aplicación Streamlit
st.title("Aplicación de Speech to Text + Insight")
st.write("Sube un archivo de audio para transcribirlo a texto.")

# Subida de archivo de audio
archivo_audio = st.file_uploader("Selecciona un archivo de audio", type=["mp3", "wav"])

# Transcribir audio si se ha subido un archivo
if archivo_audio:
    st.audio(archivo_audio, format="audio/wav")  # Reproducir el archivo de audio
    texto_transcrito = None
    if st.button("Transcribir"):
        texto_transcrito = transcribir_audio(archivo_audio)
        # st.write("Texto transcribido:")
        # st.write(texto_transcrito)
        st.session_state.texto_transcrito = texto_transcrito

# Mostrar texto transcribido si está disponible en el estado de la sesión
if "texto_transcrito" in st.session_state:
    st.write("Texto transcribido:")
    st.write(st.session_state.texto_transcrito)

if "texto_transcrito" in st.session_state:
    if st.button("Resumir"):
        texto_resumen = resumen_audio(st.session_state.texto_transcrito)
        # st.write("Texto resumen:")
        # st.write(texto_resumen)
        st.session_state.texto_resumen = texto_resumen

# Mostrar texto resumido si está disponible en el estado de la sesión
if "texto_resumen" in st.session_state:
    st.write("Texto resumen:")
    st.write(st.session_state.texto_resumen)

#------------------------------------------
# streamlit run app_speech.py  --theme.base "dark"