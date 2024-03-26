from serpapi import GoogleSearch
import os
#from dotenv import load_dotenv
import requests
from token_controler import *



#load_dotenv()



def info_extraction(subject, api_key, length=100, min_search=20):
    if not api_key:
        raise ValueError("Google Search API key is required.")

    # Fetch summary from the internet using the info_extraction function
    params = {
        "engine": "google",
        "q": subject,
        "api_key": api_key
    }

    response = requests.get("https://serpapi.com/search", params=params)

    if response.status_code == 200:
        results = response.json()
        organic_results = results.get("organic_results", [])

        # Initialize summary list with rank
        summary = []
        for result in organic_results[:min_search]:
            snippet = result.get('snippet', '')
            source_url = result.get('link', '')
            words = snippet.split()[:length]  # Select the first 'length' words
            truncated_snippet = ' '.join(words)

            # Append the summary with metadata including rank
            summary.append({
                "snippet": truncated_snippet,
                "source": source_url
            })

        return summary
    else:
        raise Exception(f"Failed to fetch search results. Status code: {response.status_code}")


if __name__ == "__main__":
    serper_ai_key = "" ""
    headline = "It's a headline"
    headline = limit_tokens(headline)
    context = info_extraction(headline, serper_ai_key)