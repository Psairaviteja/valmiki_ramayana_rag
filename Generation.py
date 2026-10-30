# from app import response
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from Rretreive import retreiver

def generate_answer(user_query:str,retrieved_docs:list) -> str:
    """
    Take Retrieved Docs and User Query and return a refined answer using LLM
    """

    if not retrieved_docs:
        return("Could not find any docs retrieved relevent to the user query")

    context = context = "\n\n---\n\n".join(
        doc.page_content for doc in retrieved_docs
    )

    system_prompt = f"""You are a helpful assistant answering questions
    about the Valmiki Ramayana.

    Use ONLY the context provided below.

    Rules:
    1. Answer clearly and concisely.
    2. If the context does not contain the answer, say:
    "The retrieved Ramayana passages do not contain enough information to answer this."
    3. Do not use outside knowledge.
    4. Do not invent names, events, or quotations.

    Context:
    {context}
    """

    messages = [SystemMessage(content = system_prompt),
    HumanMessage  (content = user_query)]

    llm = ChatOllama(model = "deepseek-r1:1.5b ", temperature = 0.0)

    response = llm.invoke(messages)

    return response.content

if __name__ == "__main__":
    user_query = "In the battle how many people did Lakshmana kill"
    retrieved_docs = retreiver(user_query)
    response = generate_answer(user_query=user_query,retrieved_docs=retrieved_docs)
    print("Chunks : ",retrieved_docs)
    print("LLM Response",response)