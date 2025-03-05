from flask import Flask, request, jsonify, render_template, Response, stream_with_context
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import Tool, initialize_agent, AgentType
from nltk.tokenize import sent_tokenize
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)

# Initialize LLM with streaming capability
llm = ChatOpenAI(temperature=0.7, streaming=True)

search = GoogleSerperAPIWrapper()
search_tool = Tool(
    name="Google Search",
    func=search.run,
    description="Useful for searching the internet for content ideas and information."
)

agent = initialize_agent(
    [search_tool],
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

idea_prompt = PromptTemplate(
    input_variables=["topic"],
    template="Generate 5 blog post ideas about {topic}. Return only the numbered list of ideas."
)

article_prompt = PromptTemplate(
    input_variables=["topic", "research"],
    template="Write a blog article about {topic} using the following research: {research}. The article should be up to 1000 words."
)

rewrite_prompts = {
    "standard": PromptTemplate(
        input_variables=["sentence"],
        template="Rewrite the following sentence while maintaining its original meaning: {sentence}"
    ),
    "creative": PromptTemplate(
        input_variables=["sentence"],
        template="Creatively rewrite the following sentence, adding more descriptive language: {sentence}"
    ),
    "formal": PromptTemplate(
        input_variables=["sentence"],
        template="Rewrite the following sentence in a more formal tone: {sentence}"
    ),
    "casual": PromptTemplate(
        input_variables=["sentence"],
        template="Rewrite the following sentence in a casual, conversational tone: {sentence}"
    )
}

idea_chain = LLMChain(llm=llm, prompt=idea_prompt)
article_chain = LLMChain(llm=llm, prompt=article_prompt)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate_ideas', methods=['POST'])
def generate_ideas():
    topic = request.form['topic']
    return Response(stream_with_context(stream_ideas(topic)), content_type='text/event-stream')

def stream_ideas(topic):
    for chunk in idea_chain.stream({"topic": topic}):
        yield f"data: {chunk}\n\n"
    yield "data: [DONE]\n\n"

@app.route('/generate_article', methods=['POST'])
def generate_article():
    topic = request.form['topic']
    research = agent.run(f"Find information about {topic} for a blog article. Include at least 3 relevant facts or statistics.")
    return Response(stream_with_context(stream_article(topic, research)), content_type='text/event-stream')

def stream_article(topic, research):
    for chunk in article_chain.stream({"topic": topic, "research": research}):
            yield f"data: {chunk}\n\n"
    yield "data: [DONE]\n\n"

@app.route('/rewrite_text', methods=['POST'])
def rewrite_text():
    text_to_rewrite = request.form['text_to_rewrite']
    mode = request.form['mode']
    return Response(stream_with_context(stream_rewrite(text_to_rewrite, mode)), content_type='text/event-stream')

def stream_rewrite(text_to_rewrite, mode):
    sentences = sent_tokenize(text_to_rewrite)
    rewrite_prompt = rewrite_prompts[mode]
    rewrite_chain = LLMChain(llm=llm, prompt=rewrite_prompt)
    for sentence in sentences:
        #Pass a dictionary as input to stream()
        for chunk in rewrite_chain.stream({"sentence": sentence}):
            yield f"data: {chunk}\n\n"
        yield "data: \n\n"
    yield "data: [DONE]\n\n"

if __name__ == '__main__':
    app.run(debug=True)
