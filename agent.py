from langgraph.prebuilt import create_react_agent
from langchain_ollama import ChatOllama
from langchain_core.tools import tool

@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    return f"Results for {query}"

@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))

llm = ChatOllama(model="phi4-mini:3.8b")
agent = create_react_agent(llm, tools=[search_web, calculate])

result = agent.invoke({"messages": [("user", "What is the capital of France?")]})
print(result["messages"][-1].AIMessage)