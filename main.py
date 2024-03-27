import streamlit as st
from MDFEND_model import NewsClassifier
from LDA_Model import LDAModel
from langchain_community.llms import OpenAI
from OpenAI_agents import *
from token_controler import limit_tokens
from info_extraction import *
import constants

import os
from flask import Flask, request, redirect, jsonify, url_for, session, abort
from flask_cors import CORS

# ########## INIT APP ##########

# --- API Flask app ---


@st.cache_resource
def model_init(class_name):
    if class_name == "lda":
        model = LDAModel()
    elif class_name == "mdfend":
        model = NewsClassifier()
    else:
        raise ValueError(f"Invalid class name: {class_name}")
    return model


app = Flask(__name__)
app.secret_key = "super secret key"

CORS(app)


# def main():


# --- BACKEND API ---


@app.route("/")
# @app.doc(hide=True)
def index():
    """Define the content of the main fontend page of the API server"""

    return f"""
    <h1>The 'Omdena's IREX API' server is up.</h1>
    """


@app.route("/inference", methods=["POST", "GET"])
def inference():
    """Infer using the LLM model and MDFEND"""

    news = """
    General Cienfuegos furioso :" Si queremos un cambio debemos revelarnos contra EPN"
    En un vídeo difundido en redes sociales por la cadena de noticias "Cierre de Edición" se confirma la participación de miembros del Ejercito Mexicanos, quienes planean rebelarse contra el presidente de México Enrique Peña Nieto.
    El Ejercito Nacional encabezado por el General Salvador Cienfuegos Zepeda, titular de la Secretaría de la Defensa Nacional (Sedena), reconoció que la entrada de los militares a la lucha contra el narcotráfico fue un error. El Ejército, dijo, debió resolver un problema "que no nos tocaba", debido a que las corporaciones estaban corrompidas y esta acción solo les permitió ampliar la ventaja para seguir con sus actos ilícitos.
    Sumando la condición en la que se encuentra actualmente México con el aumento de la gasolina el General Cienfuegos afirmo que el ejército está a la entera y completa disposición del pueblo mexicano, "Creo que no me toca decirlo a mí, pero las encuestas lo dicen, este gobierno está perdido, solo falta la decisión del pueblo y el ejercito los respaldara y protegerá para hacer cumplir sus derechos constitucionales." asimismo añadió "Si queremos un cambio vamos a rebelarnos en contra de Peña Nieto
    """

    if request.method == "POST":
        headline = request.form.get("headline")
        news = request.form.get("news")

        headline = limit_tokens(headline)
        news = limit_tokens(news)
        if True:
            lda_label = lda_model.predict_topic(news)
            result_pred_proba = model.predict(news, lda_label)
            st.write("ML model output", 100 * result_pred_proba)
            class_result = class_agent.run_class_agent(headline=headline)
            st.write("Class result:", class_result)  # Modification here
            data = json.loads(class_result)
            subject = data["subject"]
            event = data["event"]
            print(headline)
            # Info Extraction
            context = info_extraction(headline, serper_ai_key)
            st.write("Contexto-Fuentes utilizadas:", context)  # Modification here

            # Headline Alignment
            alignment_result = headline_agent.analyze_alignment(
                headline=headline, news=news
            )
            st.write(
                "Alineamiento Titular-Noticia:", alignment_result
            )  # Modification here
            data_alignment = json.loads(alignment_result)
            print(alignment_result)
            alignment_label = data_alignment["label"]

            # Misinformation Campaign Filter
            filter_result = filter_agent.run_filter_agent(headline, context)
            st.write("Difusion del Titular (times)", filter_result)  # Modification here
            filter_data = json.loads(filter_result)
            times = filter_data["times"]

            # Display Filter results

            # Decision Making Agent
            decision_result = decision_agent.run_decision_agent(
                news, context, result_pred_proba, alignment_label, times
            )

            # Final decision display
            return decision_result
    else:
        return f"""
    <h1>Inference server</h1>
    """


# ########## START FLASK SERVER ##########

if __name__ == "__main__":
    os.environ["OPENAI_API_KEY"] = constants.OPENAI_API_KEY
    os.environ["SERPER_API_KEY"] = constants.SERPER_API_KEY

    openai_api_key = constants.OPENAI_API_KEY
    serper_ai_key = constants.SERPER_API_KEY

    lda_model = model_init("lda")
    model = model_init("mdfend")

    client = OpenAI(temperature=0)
    # Initialize agents
    class_agent = ClassAgent(client=client)
    headline_agent = HeadlineAgent(client=client)
    filter_agent = FilterAgent(client=client)
    decision_agent = DecisionAgent(client=client)

    search_agent = InfoExtraction(serper_ai_key)

    current_port = int(os.environ.get("PORT") or 5000)
    app.debug = True
    app.run(debug=False, host="0.0.0.0", port=current_port, threaded=True)
