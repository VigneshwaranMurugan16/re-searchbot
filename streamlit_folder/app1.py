import streamlit as st 
from langchain_core.messages import HumanMessage, AIMessage
import boto3
import os
import glob
import argparse
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_aws import ChatBedrock
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader 
from langchain_community.document_loaders import DirectoryLoader
from langchain.load import dumps, loads
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import PyPDFLoader
from typing import Literal
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema import Document

def load_all_files(directory):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        is_separator_regex=False,
        separators=["\n"]
    )
    
    # Get all .pptx files in the directory
    files = glob.glob(os.path.join(directory, "*"))
    
    value = []
    
    for file in files:
        print(file)
        file_type = file.split('.')
        splits = []
        if file_type[-1] == 'pdf':
            print('pdf')
            pdf_loader = PyPDFLoader(file)
            documents = pdf_loader.load()
            splits = text_splitter.split_documents(documents)
        elif file_type[-1] == 'pptx':
            print('pptx')
            loader = UnstructuredPowerPointLoader(file)
            content = loader.load()
            splits = text_splitter.split_documents(content)
        elif file_type[-1] == 'docx':
            print('docx')
            loader = UnstructuredWordDocumentLoader(file)
            data = loader.load()
            splits = text_splitter.split_documents(content)
        else:
            continue
        value.extend(splits)
    return value

def format_Tdocs(docs, k=10):
    return "\n\n".join(f'This Extract is from: {doc[0].metadata["source"]} \n The content of the extract: {doc[0].page_content}' for doc in docs[:k+1])

def createQuestionsList(text):
    return text.split("\n")

def reciprocal_rank_fusion(results, k=60):
    """ Reciprocal_rank_fusion that takes multiple lists of ranked documents 
        and an optional parameter k used in the RRF formula """
    
    # Initialize a dictionary to hold fused scores for each unique document
    fused_scores = {}

    # Iterate through each list of ranked documents
    for docs in results:
        # Iterate through each document in the list, with its rank (position in the list)
        for rank, doc in enumerate(docs):
            # Convert the document to a string format to use as a key (assumes documents can be serialized to JSON)
            doc_str = dumps(doc)
            # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            # Retrieve the current score of the document, if any
            previous_score = fused_scores[doc_str]
            # Update the score of the document using the RRF formula: 1 / (rank + k)
            fused_scores[doc_str] += 1 / (rank + k)

    # Sort the documents based on their fused scores in descending order to get the final reranked results
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    # Return the reranked results as a list of tuples, each containing the document and its fused score
    return reranked_results

def Setup(data_path1,vb_path1):
    print('hello, setup starts')
    load_dotenv()
    GROQ_API_KEY = os.getenv('GROQ_API_KEY')
    
    llm = ChatGroq(model="llama3-8b-8192", temperature=0)
    
    splits = load_all_files(data_path1)
    
    model_name = "BAAI/bge-small-en-v1.5"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    hf_embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs, show_progress=True
    )
    
    FAISS_PATH = os.path.join(os.getcwd(), vb_path1)
    vectorstore = FAISS.load_local(FAISS_PATH, hf_embeddings, allow_dangerous_deserialization = True)
    retriever = vectorstore.as_retriever()
    
    bm25_retriever = BM25Retriever.from_documents(splits)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, retriever], weights=[0.5, 0.5]
    )
    
    return llm, retriever, ensemble_retriever

def Get_Reponse(user_query, chat_history, llm, retriever, ensemble_retriever):
    
    #Getting Summary Block.
    summary_template = """Imagine you are a research who is researching for your master's paper and you are supposed to provide a set of descriptive topics headings that are provided within the document.

    Example: Deep Learning For Breast Cancer, ReNest Usecase in Medical Field...

    Provide just the topics seperated by commas without any other extra text as i will send this output directly to another llm.

    Content: {content}
    """
    summary_prompt = ChatPromptTemplate.from_template(summary_template)
    summary_rag_chain = (
        {"content": ensemble_retriever,
        "question": RunnablePassthrough()} 
        | summary_prompt
        | llm
        | StrOutputParser()
    )
    summary = summary_rag_chain.invoke("provide core topics explained within the document.")
    
    #Checking for WBS or VBS. 
    class RouteQuery(BaseModel):
        datasource: Literal["vectorstore", "web_search",] = Field(
            description="Given a user question choose to route it to web search or a vectorstore.",
        )
    structured_llm_router = llm.with_structured_output(RouteQuery)
    system = f"""You are an expert at routing a user question to a vectorstore or web search.
    The summary of the document is: {summary}
    Use the vectorstore for questions that can't be answered within the indepth information provided within the summary. Otherwise, use web-search.
    Please note that the user might specifically ask to work with vectorstore/documents or web-search/internet"""
    route_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{question}"),
        ]
    )
    question_router = route_prompt | structured_llm_router
    question_router_value = question_router.invoke({"question": user_query})
    
    # All Routes Uses.
    final_template = """Imagine you are a research who is researching for your master's paper and you come across a paper which works in the same idea as yours. 
    You have a few questions which you need answer. So you are writting down the questions and looking for the answers in the paper's contents. 

    Please be precise and short with your answer and don't provide additional text that doesn't directly contribute to the answer.
    
    Along with that, Previous Chat History is also provided for better understanding which has both the ai and human responses. 
    Focus mainly on the question and content; involve the chat history for references alone.

    Questions: {question}

    Content: {content}
    
    chat_history: {chat_history} 
    """
    final_prompt = ChatPromptTemplate.from_template(final_template)
    
    # Question Routed to VectorStore.
    if question_router_value.datasource == 'vectorstore':
        print("Running into Vectorstore.")
        # Multi Question Generation Setup.
        mulit_question_template = """You are an AI language model assistant. Your task is to generate five
        different versions of the given user question to retrieve relevant documents from a vector
        database. By generating multiple perspectives on the user question, your goal is to help
        the user overcome some of the limitations of the distance-based similarity search.
        Provide these alternative questions separated by newlines. Original question: {question}

        Since there are some infomation that should be added to the question in order to provide the best results to retrieve the documents from the vector database. 
        You are supposed to include technical keywords to further help the document retrieval process.
        As the LLM which will analysis these questions and the retrieved documents have no prior knowledge on the documents nor the question, make the question technical

        NOTE: Be Direct with your questions"""
        prompt_rag_fusion = ChatPromptTemplate.from_template(mulit_question_template)

        getMoreQuestions = ({"question": RunnablePassthrough()} 
                            | prompt_rag_fusion 
                            | ChatGroq(model="mixtral-8x7b-32768")  
                            | StrOutputParser()  
                            | createQuestionsList)
        retrieval_chain_rag_fusion = getMoreQuestions | ensemble_retriever.map() | reciprocal_rank_fusion
        
        # Question being checked if it's relevent. 
        class GradeDocuments(BaseModel):
            binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")
        structured_llm_grader = ChatGroq(temperature=0).with_structured_output(GradeDocuments)

        # Prompt 
        system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
            If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
            It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
        grade_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
            ]
        )

        retrieval_grader = grade_prompt | structured_llm_grader
        
        class GradeHallucinationsAndAnswer(BaseModel):
            """Binary score for hallucination present in generation answer and to assess answer addresses question."""

            binary_score: str = Field(description="Answer is grounded in the facts with no hallucinations and completely answers the question, 'yes' or 'no'")
            
        structured_llm_grader = ChatGroq(temperature=0).with_structured_output(GradeHallucinationsAndAnswer)
        
        system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts and assessing whether an answer addresses / resolves a question  \n 
            Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts and that the answer resolves the question."""
            
        hallucination_and_answer_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
            ]
        )

        hallucination_and_answer_grader = hallucination_and_answer_prompt | structured_llm_grader
        
        # Loops till you get the correct documents.
        while True: 
            while True:
                print('Running into Question Checker.')
                
                docs = retrieval_chain_rag_fusion.invoke(user_query)
                doc_txt = format_Tdocs(docs)
                doc_grads = retrieval_grader.invoke({"question": user_query, "document": doc_txt})
                if doc_grads.binary_score != 'yes':
                    print('Given Question is wrong.')
                    #rewritting the user query.
                    system = """You a question re-writer that converts an input question to a better version that is optimized \n 
                        for vectorstore retrieval. Look at the input and try to reason about the underlying sematic intent / meaning. Just provide the new question alone no need for any other content."""
                    re_write_prompt = ChatPromptTemplate.from_messages(
                        [
                            ("system", system),
                            ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question."),
                        ]
                    )
                    user_query = re_write_prompt | ChatGroq(temperature=0) | StrOutputParser()   
                else:
                    break
        
            # Now the rest of the complete generation:
            print('Running into Answer Checker.')
            final_rag_chain = (
                final_prompt
                | llm
                | StrOutputParser()
            )
            answer = final_rag_chain.invoke({"content": doc_txt,"question": user_query, "chat_history": chat_history})
        
            # Now testing the hallucinations and checking if the answer is correct
            HAOutput = hallucination_and_answer_grader.invoke({"documents": doc_txt, "generation": answer})
            # print(answer, HAOutput)
            if HAOutput.binary_score == 'yes':
                return answer
            else:
                print('Given Answer is incorrect.')
        
    # Question Routed to web search.
    if question_router_value.datasource == 'web_search':
        print('Running Web Search')
        st.markdown(f"Please note the provided documenets and the provided questions seems to have mismatching topics. So, I shall revert to internet references. Take the answer with the grain of salt.")
        web_search_tool = TavilySearchResults(k=3)
        
        docs = web_search_tool.invoke({"query": user_query})
        web_results = "\n".join([d["content"] for d in docs])
        web_results = Document(page_content=web_results)
        
        final_template = """You are a helpful assistant. Answer the following questions along with the history of the chats so far. 
        Just to be sure, Web Search Results of the same question is also provided. Please refer to if you require it. If you do not require it, then ignore that information.
    
        Focus mainly on the question and involve the web search results and chat history for references alone.

        Questions: {question}

        Web Search Results: {content}
        
        chat_history: {chat_history} 
        """
        final_prompt = ChatPromptTemplate.from_template(final_template)
        final_rag_chain = (
                final_prompt
                | llm
                | StrOutputParser()
            )
        answer = final_rag_chain.invoke({"content": web_results,"question": user_query, "chat_history": chat_history})
        return answer
        
        
    
    
    # template = """
    # You are a helpful assistant. Answer the following questions along with the history of the chats so far. 
    
    # Focus mainly on the question and involve the chat history for references alone.
    
    # question: {question}
    
    # chat_history: {chat_history}
    # """
    # prompt1 = ChatPromptTemplate.from_template(template)
    # final_rag_chain_QA = (
    # prompt1
    # | llm
    # | StrOutputParser()
    # )
    # return final_rag_chain_QA.invoke({"question": user_query, "chat_history": chat_history})


if __name__ == '__main__':
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if 'state_setup' not in st.session_state:
        data_path1 = r'D:\My-Fast-API\uploaded_files'
        vb_path1 = r'D:\My-Fast-API\Main_Files'
        llm, retriever, ensemble_retriever = Setup(data_path1, vb_path1)
        st.session_state.state_setup = [llm, retriever, ensemble_retriever]

    st.set_page_config(page_title="RAGSync: Chat")
    st.title('RAGSync Bot')

    for message in st.session_state.chat_history: 
        if isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.markdown(message.content)
        else:
            with st.chat_message('YBot'):
                st.markdown(message.content)

    user_query = st.chat_input("Your query")
    if user_query is not None and user_query != '':
        st.session_state.chat_history.append(HumanMessage(user_query))
    
        with st.chat_message('Human'):
            st.markdown(user_query)
        
        with st.chat_message("YBot"):
            ai_message = Get_Reponse(user_query, st.session_state.chat_history, st.session_state.state_setup[0], st.session_state.state_setup[1], st.session_state.state_setup[2])
            st.markdown(ai_message)
            st.session_state.chat_history.append(AIMessage(ai_message))