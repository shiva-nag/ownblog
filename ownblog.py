from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
from langchain.agents import Tool, initialize_agent, AgentType
# from langchain_community.llms import OpenAI
# from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOpenAI
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from nltk.tokenize import sent_tokenize

load_dotenv()

app = Flask(__name__)

# Initialize LLM
#llm = OpenAI(temperature=0.7)
llm = ChatOpenAI(temperature=0.5, model_name="gpt-3.5-turbo")

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
    input_variables=["topic", "research", "section"],
    template="Write a {section} about {topic} using the following research: {research}. The {section} should be up to 100 words."
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

# Chains
#idea_chain = LLMChain(llm=llm, prompt=idea_prompt)
idea_chain = idea_prompt | llm
article_chain = article_prompt | llm
#article_chain = LLMChain(llm=llm, prompt=article_prompt)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate_ideas', methods=['POST'])
def generate_ideas():
    topic = request.json['topic']
    ideas = idea_chain.invoke(topic)
    return jsonify({"ideas": ideas})

@app.route('/generate_article', methods=['POST'])
def generate_article():
    topic = request.json['topic']
    research = agent.run(f"Find information about {topic} for a blog article. Include at least 3 relevant facts or statistics. Include citations for these statistics")
    sections = [
        "introduction",
        "body paragraph 1",
        "body paragraph 2",
        "body paragraph 3",
        "body paragraph 4",
        "concluding summary"
    ]

    article_sections = {}
    for section in sections:
#        section_chain = LLMChain(llm=llm, prompt=article_prompt)
        section_content = article_chain.run(topic=topic, research=research, section=section)
        article_sections[section] = section_content

    full_article = "\n".join(article_sections.values())
    return render_template('index.html', topic=topic, article=full_article)
#    return jsonify({"article": full_article})


@app.route('/rewrite_text', methods=['POST'])
def rewrite_text():
    text_to_rewrite = request.form['text_to_rewrite']
    mode = request.form['mode']

    sentences = sent_tokenize(text_to_rewrite)

    rewritten_sentences = []
    for sentence in sentences:
        rewrite_prompt = rewrite_prompts[mode]
        rewrite_chain = LLMChain(llm=llm, prompt=rewrite_prompt)
        rewritten_sentence = rewrite_chain.run(sentence=sentence)
        rewritten_sentences.append(rewritten_sentence)

    rewritten_text = " ".join(rewritten_sentences)

    return render_template('index.html', rewritten_text=rewritten_text)

if __name__ == '__main__':
    app.run(debug=True)

