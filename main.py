import streamlit as st
from MDFEND_model import NewsClassifier
from LDA_Model import LDAModel

# Press the green button in the gutter to run the script.


def main():
    # Text input box
    input_text = st.text_area('Enter text here:')

    lda_model = LDAModel()
    model = NewsClassifier()
    # result = model.predict('text', lda_label)

    if st.button('Process'):
        if input_text:
            lda_label = lda_model.predict_topic(input_text)
            result = model.predict(input_text, lda_label)
            st.write('Processed text:', result)
        else:
            st.warning('Please enter some text.')


if __name__ == '__main__':
    main()
