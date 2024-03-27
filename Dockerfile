FROM python:3.10.12-slim

ENV PORT=5000
EXPOSE 5000

# --- start to install backend-end stuff
RUN mkdir -p /app
RUN mkdir -p /models
WORKDIR /app

# --- Install Python requirements.
RUN apt-get update
RUN apt-get install -y python3
RUN apt-get install -y python3-pip

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# --- Copy project files
COPY ["main.py", "./"]
COPY ["models", "./models/"]
COPY ["constants.py", "./"]
COPY ["token_controler.py", "./"]
COPY ["OpenAI_agents.py", "./"]
COPY ["MDFEND_model.py", "./"]
COPY ["LDA_Model.py", "./"]
COPY ["info_extraction.py", "./"]

# --- Start server
CMD gunicorn main:app --bind 0.0.0.0:$PORT --timeout=600 --threads=10
# CMD ["python", "main.py"]