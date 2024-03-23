from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import OpenAI
#from langchain.llms import OpenAI
import json
import serpapi
import os


class FakeDetectionWrapper:
    def __init__(self, client):
        self.client = client


class FilterAgent(FakeDetectionWrapper):
    def __init__(self, client):
        super().__init__(client)

        self.filter_agent = """ You are an agent that must label the subjectivity of news {news}, if the news is a 
        personal opinion, impossible to verify (1), ifthe news is an objective statement (0) if the statement is about
        an event or fact that can potentially be verified with evidence, even if the evidence is not currently 
        available in the news {news} provided.it expresses a personal opinion or cannot be verified objectively.

        present the label  in a JSON structure
        (
          "label": label,
        )
        """
        self.prompt_filter_agent = PromptTemplate(template=self.filter_agent, input_variables=["news"])
        self.llm_chain_filter_agent = LLMChain(prompt=self.prompt_filter_agent, llm=self.client)

    def run_filter_agent(self, news):
        try:
            filter_layer_output = self.llm_chain_filter_agent.run({'news': news})
            return filter_layer_output
        except Exception as e:
            print(e)
            return "Error filter layer"


class ClassAgent(FakeDetectionWrapper):
    def __init__(self, client):
        super().__init__(client)

        self.class_agent = """ You are an agent with the task of identifying the elements subject of a news {news} .
        you will identify the subject, the event and the field the news belongs to Politics, Economics and Social.
        you will provide a Json Structure :
          (
          "subject":  subject of the news,
          "event": event described,
          "topic": field the news belongs to (Politics, Economics or Social
        )
        """
        self.prompt_class_agent = PromptTemplate(template=self.class_agent, input_variables=["news"])
        self.llm_chain_class_agent = LLMChain(prompt=self.prompt_class_agent, llm=self.client)

    def run_class_agent(self, news):
        try:
            class_agent_output = self.llm_chain_class_agent.run({'news': news})
            return class_agent_output
        except Exception as e:
            print(e)
            return "Error class layer"


class DecisionAgent(FakeDetectionWrapper):
    def __init__(self, client):
        super().__init__(client)
        self.decision_agent = """You will be presented with a piece of news {news} and contextual information 
            gathered from the internet {context}.
            Your task is to evaluate whether the {news} is genuine or not, based solely on:

            - Correlation with information from external sources {context},
            - The likelihood of it being false as determined by an ML algorithm {probability}, if it is higher than 0.6 
            it is highly likely to be fake.

            Based on these criteria, you must decide if the evidence supports the authenticity of the news. Your 
            conclusion should include:

            "category": A label indicating whether the news is fake or real,
            "reasoning": A detailed explanation supporting your classification.
            Provide it in Spanish.
            """
        self.prompt_decision_agent = PromptTemplate(template=self.decision_agent,
                                                    input_variables=["news", "context", "probability"])

        # Corrected the LLMChain instantiation
        self.llm_chain_decision_agent = LLMChain(prompt=self.prompt_decision_agent, llm=self.client)

    def run_decision_agent(self, news, context, probability):
        try:
            decision_agent_output = self.llm_chain_decision_agent.run(
                {'news': news, 'context': context, 'probability': probability})
            return decision_agent_output
        except Exception as e:
            print(e)
            return "Error decision layer"


class SummaryAgent(FakeDetectionWrapper):
    def __init__(self, client):
        super().__init__(client)

        self.summary_agent = """Make a summary of the input {RAG_input} in 200 words, discard contradicting info and 
        narrow it down to stable and verifiable data, avoid contradictory information in the text
        """
        self.prompt_summary_agent = PromptTemplate(template=self.summary_agent, input_variables=["RAG_input"],
                                                   output_key="summary")
        # Corrected the LLMChain instantiation
        self.llm_chain_summary_agent = LLMChain(prompt=self.prompt_summary_agent, llm=self.client)

    def run_summary_agent(self, RAG_input):
        try:
            summary_agent_output = self.llm_chain_summary_agent.run({'RAG_input': RAG_input})
            return summary_agent_output
        except Exception as e:
            print(e)
            return "Error summary layer"


class InfoExtraction:
    def __init__(self, api_key=None):
        self.serper_ai_key = api_key

        # Define the mapping of topics to their prioritized sources
        self.topic_priority_map = {
            "Politics": {
                "POLITICAL PARTIES": {"name": "ACCIÓN CIUDADANA", "link": "https://accion-ciudadana.org/"},
                "TRANSPARENCY": {"name": "TRACODA", "link": "https://tracoda.info/"},
                "ELECTIONS": {"name": "VOTANTE", "link": "https://twitter.com/somosvotante"},
                "CORRUPTION": {"name": "ALAC", "link": "https://twitter.com/ALAC_SV"},
            },
            "Social": {
                "GENDER": {"name": "ORMUSA", "link": "https://ormusa.org/"},
                "VIOLENCE": {"name": "ASDEUH", "link": "https://asdehu.com/"},
                "ENVIRONMENT": {"name": "ACUA", "link": "https://www.acua.org.sv/"},
                "MIGRATION": {"name": "GMIES", "link": "https://gmies.org/"},
            },
            "Economy": {
                "BUDGET": {"name": "FUNDE", "link": "https://funde.org/"},
                "MACROECONOMY": {"name": "ICEFI", "link": "https://mail.icefi.org/etiquetas/el-salvador"},
            }
        }

    def _fetch_summary(self, subject, event, topic, length, min_search):
        # Fetch summary from the internet using the info_extraction function
        params_dic = {
            "engine": "google",
            "q": subject + event,
            "api_key": self.serper_ai_key
        }

        search_result = serpapi.search(params_dic)
        #results = search_result.get_dict()
        #organic_results = results["organic_results"]
        organic_results = search_result["organic_results"]

        # Flatten all priority sources for secondary ranking
        all_priority_sources = {info['link'] for _, approaches in self.topic_priority_map.items() for _, info in
                                approaches.items()}
        topic_linked_sources = [info['link'] for approach, info in self.topic_priority_map.get(topic, {}).items()]

        # Initialize summary list with rank
        summary = []
        for result in organic_results[:min_search]:
            snippet = result['snippet']
            source_url = result['link']
            words = snippet.split()[:length]  # Select the first 'length' words
            truncated_snippet = ' '.join(words)

            # Rank determination
            if source_url in topic_linked_sources:
                rank = 1  # Highest priority: Directly linked to the topic
            elif source_url in all_priority_sources:
                rank = 2  # Secondary priority: In the priority map but not directly linked to the topic
            else:
                rank = 3  # Lowest priority: Not in the priority map

            # Append the summary with metadata including rank
            summary.append({
                "snippet": truncated_snippet,
                "source": source_url,
                "rank": rank
            })

        # Sort the summary list by 'rank'
        sorted_summary = sorted(summary, key=lambda x: x['rank'])
        return sorted_summary

    def extract_info(self, subject, event, topic, length=50, min_search=20):
        return self._fetch_summary(subject, event, topic, length, min_search)


# Example usage:
if __name__ == "__main__":
    open_ai_key = " "
    serper_ai_key = " "
    news = "It's a Fake news here"

    llm = OpenAI(temperature=0)
    filter_agent = FilterAgent(llm)
    class_agent = ClassAgent(llm)
    decision_agent = DecisionAgent(llm)
    search_agent = InfoExtraction(serper_ai_key)
    summary_agent = SummaryAgent(llm)

    output_filter = filter_agent.run_filter_agent(news)
    output_class = class_agent.run_class_agent(news)
    output_decision = decision_agent.run_decision_agent(news=news, context='no_context', probability='0.5')
    output_class_dict = json.loads(output_class)

    output_search = search_agent.extract_info(output_class_dict['subject'], output_class_dict['event'], output_class_dict['topic'], 5, 3)
    output_summary = summary_agent.run_summary_agent(output_search)

    ## Documentation https://serpapi.com/search-api
    #params = {
    #    "engine": "google",
    #    "q": 'event',
    #    "api_key": serper_ai_key
    #}

    #search = serpapi.search(params)
    with open("outputs_open_ai.txt", "w") as file:
        file.write(f"output_filter: {output_filter}\n")
        file.write(f"output_class: {output_class}\n")
        file.write(f"output_decision: {output_decision}\n")
        file.write(f"output_search: {output_search}\n")
        file.write(f"output_summary: {output_summary}\n")

    print("Data has been written to outputs.txt")
