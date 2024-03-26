import streamlit as st
from MDFEND_model import NewsClassifier
from LDA_Model import LDAModel
from langchain_community.llms import OpenAI
from OpenAI_agents import *
from token_controler import limit_tokens
from info_extraction import *

@st.cache_resource
def model_init(class_name):
    if class_name == 'lda':
        model = LDAModel()
    elif class_name == 'mdfend':
        model = NewsClassifier()
    else:
        raise ValueError(f"Invalid class name: {class_name}")
    return model


def main():
    # Text input box
    input_text = st.text_area('Enter text here:')
    open_ai_key = st.secrets["open_ai_key"]
    serper_ai_key = st.secrets["serper_ai_key"]
    os.environ["OPENAI_API_KEY"] = open_ai_key

    lda_model = model_init('lda')
    model = model_init('mdfend')

    client = OpenAI(temperature=0)
    # Initialize agents
    class_agent = ClassAgent(client=client)
    headline_agent = HeadlineAgent(client=client)
    filter_agent = FilterAgent(client=client)
    decision_agent = DecisionAgent(client=client)

    search_agent = InfoExtraction(serper_ai_key)

    # Streamlit UI Code
    st.title("Analisis de Veracidad de Noticias ")
    st.title("El Salvador")

    # Sample input for demonstration
    headline = st.text_input("Titular", "")
    news = st.text_area("Cuerpo de la Noticia", "")

    headline = limit_tokens(headline)
    news = limit_tokens(news)


    if st.button('Process'):
        if input_text:
            lda_label = lda_model.predict_topic(input_text)
            result_pred_proba = model.predict(input_text, lda_label)

            class_result = class_agent.classify(headline=headline)
            st.write("Class result:", class_result)  # Modification here
            data = json.loads(class_result)
            subject = data["subject"]
            event = data["event"]

            # Info Extraction
            context = info_extraction(serper_ai_key, headline)
            st.write("Contexto-Fuentes utilizadas:", context)  # Modification here

            # Headline Alignment
            alignment_result = headline_agent.analyze_alignment(headline=headline, news=news)
            st.write("Alineamiento Titular-Noticia:", alignment_result)  # Modification here
            data_alignment = json.loads(alignment_result)
            alignment_label = data_alignment["label"]

            # Misinformation Campaign Filter
            filter_result = filter_agent.filter_context(headline, context)
            st.write("Difusion del Titular (times)", filter_result)  # Modification here
            filter_data = json.loads(filter_result)
            times = filter_data["times"]

            # Display Filter results

            # Decision Making Agent
            decision_result = decision_agent.make_decision(news, context, result_pred_proba, alignment_label, times)

            # Final decision display
            st.write("Decision Final", decision_result)  # Modification here


if __name__ == '__main__':
    main()
