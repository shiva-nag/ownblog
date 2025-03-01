#
import os
from flask import Flask, request, jsonify, render_template
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.agents import Tool, initialize_agent, AgentType

app = Flask(__name__)

# Set up API keys (replace with your actual keys)
os.environ["OPENAI_API_KEY"] = "your_openai_api_key"
os.environ["SERPER_API_KEY"] = "your_serper_api_key"

# Initialize LLM
llm = OpenAI(temperature=0.7)

# Set up Google Search
search = GoogleSerperAPIWrapper()
search_tool = Tool(
    name="Google Search",
    func=search.run,
    description="Useful for searching the internet for content ideas and information."
)

# Initialize agent
agent = initialize_agent(
    [search_tool],
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Prompt templates
idea_prompt = PromptTemplate(
    input_variables=["topic"],
    template="Generate 5 blog post ideas about {topic}. Return only the numbered list of ideas."
)

article_prompt = PromptTemplate(
    input_variables=["topic", "research"],
    template="Write a blog article about {topic}. Use the following research to include citations: {research}. The article should have an introduction, body paragraphs, and a conclusion. Include at least 3 citations in the format [1], [2], etc."
)

# Chains
idea_chain = LLMChain(llm=llm, prompt=idea_prompt)
article_chain = LLMChain(llm=llm, prompt=article_prompt)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate_ideas', methods=['POST'])
def generate_ideas():
    topic = request.json['topic']
    ideas = idea_chain.run(topic)
    return jsonify({"ideas": ideas})

@app.route('/generate_article', methods=['POST'])
def generate_article():
    topic = request.json['topic']
    research = agent.run(f"Find information about {topic} for a blog article. Include at least 3 relevant facts or statistics.")
    article = article_chain.run(topic=topic, research=research)
    return jsonify({"article": article})

if __name__ == '__main__':
    app.run(debug=True)

