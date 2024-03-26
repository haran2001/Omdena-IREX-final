from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import OpenAI
#from langchain.llms import OpenAI
import json
import serpapi
import os


class FilterAgent:
    def __init__(self, client):
        self.client = client
        filter_agent_template = """ You are an agent with the task of counting in how many entries of a context {context},
                a text extract {headline} can be found identically and literally word by word.
                You will review each entry and see if the extract {headline} can be found exactly the same within each entry,
                not just similar semantically but word by word.

                your job is to generate a JSON structure with the number of entries where this happens:

                      (
                        "times": number of entries where the headline is found exactly and literally word by word,
                      )
                """
        self.__prompt_template = PromptTemplate(template=filter_agent_template, input_variables=["headline", "context"])
        self.llm_chain = LLMChain(prompt=self.__prompt_template, llm=self.client)

    def run_filter_agent(self, headline, context):
        try:
            output = self.llm_chain.run({'headline': headline, 'context': context})
            return output
        except Exception as e:
            print(e)
            return "Error in filtering layer"


class ClassAgent:
    def __init__(self, client):
        self.client = client

        class_agent_template = """ You are an agent with the task of analysing a headline {headline} .
                you will identify the subject, the event, and the field the news belongs to either Politics, Economics, or Social.
                you will provide a JSON Structure:
                  (
                  "subject": subject of the news,
                  "event": event described,
                  "topic": field the news belongs to Politics, Economics, or Social
                  )
                """
        self.prompt_template = PromptTemplate(template=class_agent_template, input_variables=["headline"])
        self.llm_chain = LLMChain(prompt=self.prompt_template, llm=self.client)

    def run_class_agent(self, headline):
        try:
            output = self.llm_chain.run({'headline': headline})
            return output
        except Exception as e:
            print(e)
            return "Error in classification layer"


class DecisionAgent:
    def __init__(self, client):
        self.client = client
        decision_agent_template = """you are information verification agent in 2024,
                You will be presented with a piece of news {news} and information gathered from the internet {filtered_context}.
                Your task is to evaluate whether the news is real or fake, based solely on:

                - How the {news} corresponds to the information retrieved {filtered_context}, considering the reliability of the sources.
                - Probability of the news {probability} being real.
                - Alignment of the headline and the news {alignment},Not aligment is a sign of fake news .
                - Number of times the exact headline is found in other media outlets {times} which could indicate a misinformation campaign.

                Based on these criteria provided in order of importance,
                produced a reasoned argumentation whether the news is Fake or real.
                You answer strictly as a single JSON string. Don't include any other verbose texts and don't include the markdown syntax anywhere.

                  (
                "category": Fake or Real,
                "reasoning": Your reasoning here.
                   )  
                provide your answers in Spanish
                """
        self.prompt_template = PromptTemplate(template=decision_agent_template,
                                              input_variables=["news",
                                                               "filtered_context",
                                                               "probability",
                                                               "alignment",
                                                               "times"])
        self.llm_chain = LLMChain(prompt=self.prompt_template, llm=self.client)

    def run_decision_agent(self, news, filtered_context, probability, alignment, times):
        try:
            output = self.llm_chain.run(
                {'news': news, 'filtered_context': filtered_context, 'probability': probability, 'alignment': alignment,
                 'times': times})
            return output
        except Exception as e:
            print(e)
            return "Error in decision layer"


class HeadlineAgent:
    def __init__(self, client):
        self.client = client
        headline_agent_template = """ You are an agent with the task of identifying the 
        whether the headline {headline} is aligned with
        the body of the news {news}.
        you will generate a Json output:

      (
        "label": Aligned or not Aligned,
      )

        """
        self.prompt_template = PromptTemplate(template=headline_agent_template, input_variables=["headline", "news"])
        self.llm_chain = LLMChain(prompt=self.prompt_template,
                                  llm=self.client)  # Assuming LLLMChain was a typo and should be LLMChain

    def analyze_alignment(self, headline, news):
        """
        Analyzes the alignment between a given headline and the body of the news.

        Parameters:
        - headline (str): The news headline.
        - news (str): The full text of the news article.

        Returns:
        - A dictionary with the analysis results, including whether the headline is aligned with the news body and any relevant analysis details.
        """
        try:
            output = self.llm_chain.run({'headline': headline, 'news': news})
            return output
        except Exception as e:
            print(e)
            return {"error": "Error in headline alignment analysis layer"}


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


    with open("outputs_open_ai.txt", "w") as file:
        file.write(f"output_filter: {output_filter}\n")
        file.write(f"output_class: {output_class}\n")
        file.write(f"output_decision: {output_decision}\n")
        file.write(f"output_search: {output_search}\n")
        file.write(f"output_summary: {output_summary}\n")

    print("Data has been written to outputs.txt")
