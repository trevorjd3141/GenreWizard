import os
from dotenv import load_dotenv
load_dotenv()

from langchain import hub
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
# from langchain_core.messages import (
#     HumanMessage,
#     SystemMessage,
# )
from langchain_core.prompts import (
    MessagesPlaceholder,
    PromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    ChatPromptTemplate
)

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")


game = "Tekken 8"

search = TavilySearchResults()

tools = [search]
llm = ChatOpenAI(model="gpt-4o-2024-05-13", temperature=0)

system_prompt = """
"You are a helpful genre classifier that searches the web for information and then analyzes it to come to an accurate answer.

You always follow these guidelines:
  - If you do not know the answer or the game has not been released yet you will respond with -1.
  - You respond with the genre index only and nothing else.
"""
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[], template=system_prompt)),
    MessagesPlaceholder(variable_name='chat_history', optional=True),
    HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['input'], template='{input}')),
    MessagesPlaceholder(variable_name='agent_scratchpad')
])
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

genre_subset = ["Shooter", "RPG", "Arcade", "Horror", "Roguelite/Roguelike", "Real-time Strategy (RTS)", "Turn-based Strategy (TBS)", "Puzzle", "Battle Royale"]
genre_text = "\n".join([f"{i}: {genre}" for i, genre in enumerate(genre_subset)])
input = f"""
Question:
What is the number of the genre of the game \"{game}\"? Please respond with only the genre index and no other text.

The available options are:
{genre_text}
"""
agent_executor.invoke({"input": input})