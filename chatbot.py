

import os
from typing import Annotated, TypedDict
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError( "GEMINI_API_KEY not found. Check your .env file.")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=api_key)



def extract_text(content) -> str:
    """Handle string responses and lists of text blocks."""

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)

            elif isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))

            elif hasattr(item, "text"):
                parts.append(item.text)

        return "\n".join(part for part in parts if part)

    return ""



embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2")


def build_retriever(pdf_path: str):
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}. "
            "Keep the PDF in your project folder.")

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    if not documents:
        raise ValueError(f"No readable pages in {pdf_path}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100)

    chunks = splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore.as_retriever( search_kwargs={"k": 4})


academic_retriever = build_retriever("academics_handbook.pdf")

fee_retriever = build_retriever("fee_structure.pdf")


class State(TypedDict):
    programme: str
    messages: Annotated[list, add_messages]
    query_type: str
    retrieved_context: str



def classifier_node(state: State) -> dict:
    """Classify the student's latest question."""

    query = extract_text(
        state["messages"][-1].content)

    prompt = f"""
Classify this college student's question into exactly
one category: academic, fee, or general.

academic:
Attendance, exams, grades, credits, promotion,
courses, training, degree requirements.

fee:
Tuition, payments, refunds, late charges, scholarships,
or other money-related questions.

general:
Greetings, casual conversation, and other questions.

Question: {query}

Return exactly one word: academic, fee, or general.
"""

    
    response = llm.invoke(prompt)
    category = extract_text(response.content).strip().lower()

    first_word = category.split()[0].strip(".,:") if category else ""

    if first_word == "academic":
        category = "academic"
    elif first_word == "fee":
        category = "fee"
    else:
        category = "general"

    return {"query_type": category}


def academic_rag_node(state: State) -> dict:
    query = extract_text(
       state["messages"][-1].content)

    documents = academic_retriever.invoke(query)

    context = "\n\n".join( document.page_content for document in documents)

    return {"retrieved_context": context}


def fee_rag_node(state: State) -> dict:
    query = extract_text(
        state["messages"][-1].content)

    documents = fee_retriever.invoke(query)

    context = "\n\n".join( document.page_content for document in documents)

    return {"retrieved_context": context}


def general_node(state: State) -> dict:
    return { "retrieved_context": "NO_RETRIEVAL_NEEDED"}


def response_node(state: State) -> dict:
    query = extract_text(state["messages"][-1].content)

    programme = state.get("programme", "Unknown")
    context = state.get("retrieved_context", "")

    conversation_history = state["messages"][:-1]

    history_text = "\n".join(f"{message.type}: {extract_text(message.content)}" for message in conversation_history)

    if context == "NO_RETRIEVAL_NEEDED":
        prompt = f"""
You are a friendly college assistant.

The student's programme is {programme}.

Answer the student's general question naturally.
Do not invent college-specific policies or fees.

Current question:
{query}
"""

    else:
        prompt = f"""
You are a college assistant helping a {programme} student.

Answer using the official college document context below.

Rules:
- Use the supplied context for college policies and fees.
- Do not invent amounts, rules, or requirements.
- If the documents do not contain the answer, clearly
  say that the available documents do not specify it.
- If programme-specific information is available,
  prioritize the information for {programme}.
- Give a clear and friendly answer.

Official document context:
{context}

Student's question:
{query}
"""

    if history_text:
        prompt = f"""
Previous conversation:
{history_text}

Use the previous conversation to understand follow-up
questions. Answer the latest question, not an older one.

{prompt}
"""

    response = llm.invoke(prompt)
    answer = extract_text(response.content).strip()

    if not answer:
        answer = ( "Sorry, I could not generate a text response. " "Please try again.")

    return { "messages": [("ai", answer)]}



def route_query(state: State) -> str:
    category = state["query_type"]

    if category == "academic":
        return "academic_rag"

    if category == "fee":
        return "fee_rag"

    return "general"


builder = StateGraph(State)

builder.add_node("classifier", classifier_node)
builder.add_node("academic_rag", academic_rag_node)
builder.add_node("fee_rag", fee_rag_node)
builder.add_node("general", general_node)
builder.add_node("response", response_node)





builder.add_edge(START, "classifier")

builder.add_conditional_edges("classifier", route_query,{"academic_rag": "academic_rag","fee_rag": "fee_rag", "general": "general"})
builder.add_edge("academic_rag", "response")
builder.add_edge("fee_rag", "response")
builder.add_edge("general", "response")
builder.add_edge("response", END)



memory = MemorySaver()

app = builder.compile(checkpointer=memory)




def main():
    print("\nWelcome to the College Assistant!\n")

    print("Which programme are you in?")
    print("1. BCA")
    print("2. BBA")
    print("3. B.Com (H)")

    choice = input("\nEnter 1, 2 or 3: ").strip()

    programme_map = {"1": "BCA","2": "BBA","3": "B.Com (H)"}

    if choice not in programme_map:
        print("Invalid choice. Defaulting to BCA.")
        choice = "1"

    student_programme = programme_map[choice]

    print(f"\nYou are set as a {student_programme} student.")
    print("Ask your question. Type 'exit' to quit.\n")

    config = {
        "configurable": {
            "thread_id": "college_assistant"}}

    while True:
        try:
            user_query = input("You: ").strip()

            if user_query.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break

            if not user_query:
                continue

            result = app.invoke(
                {
                    "programme": student_programme,
                    "messages": [
                        ("human", user_query) ]},config=config)

            print(
                "Assistant:",
                extract_text(result["messages"][-1].content),"\n" )

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

        except Exception as error:
            print(f"\nAn error occurred: {error}")
            print("Check the error details and try again.\n")


if __name__ == "__main__":
    main()
