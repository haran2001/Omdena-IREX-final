import streamlit as st
from MDFEND_model import NewsClassifier
from LDA_Model import LDAModel
from langchain_community.llms import OpenAI
from OpenAI_agents import *
# It's a wrapper to cashing models
@st.cache_resource
def model_init(class_name):
    if class_name == 'lda':
        model = LDAModel()
    elif class_name == 'mdfend':
        model = NewsClassifier()
    else:
        raise ValueError(f"Invalid class name: {class_name}")
    return model

### Actions
def click_button():
    pass



def main():
    # Text input box
    input_text = st.text_area('Enter text here:')
    #news = "It's a fake news here"
    open_ai_key = st.secrets["open_ai_key"]
    serper_ai_key = st.secrets["serper_ai_key"]
    os.environ["OPENAI_API_KEY"] = open_ai_key

    lda_model = model_init('lda')
    model = model_init('mdfend')

    llm = OpenAI(temperature=0)
    filter_agent = FilterAgent(llm)
    class_agent = ClassAgent(llm)
    decision_agent = DecisionAgent(llm)
    search_agent = InfoExtraction(serper_ai_key)
    summary_agent = SummaryAgent(llm)


    # result = model.predict('text', lda_label)

    if st.button('Process'):
        if input_text:
            lda_label = lda_model.predict_topic(input_text)
            result_pred_broba = model.predict(input_text, lda_label)
            output_filter = filter_agent.run_filter_agent(input_text)
            output_class = class_agent.run_class_agent(input_text)
            output_decision = decision_agent.run_decision_agent(news=input_text, context='no_context', probability=result_pred_broba)
            output_class_dict = json.loads(output_class)

            output_search = search_agent.extract_info(output_class_dict['subject'], output_class_dict['event'],
                                                      output_class_dict['topic'], 5, 3)
            output_summary = summary_agent.run_summary_agent(output_search)
            st.write('The news is Fake with probability:', result_pred_broba)
            st.write('output_filter', output_filter)
            st.write('output_class', output_class)
            st.write('output_decision', output_decision)
            st.write('output_search', output_search)
            st.write('output_summary', output_summary)
        else:
            st.warning('Please enter some text.')


if __name__ == '__main__':
    main()
