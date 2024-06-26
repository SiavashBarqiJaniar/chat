


import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
# WebBaseLoader requires beautifulsoup4 to be installed
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# for embedding and vectorization:
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

def get_response(user_input):
    # invoke function replaces those variables in prompt in function get_conversational_rag_chain
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })

    return response['answer']

def get_vectorstore_from_url(url, OPENAI_API_KEY):
    """ get the textin document form"""
    loader = WebBaseLoader(url)
    document = loader.load()

    # split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)

    # create a vectorstore from the chunks
    vector_store = Chroma.from_documents(document_chunks, OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY))

    return vector_store

def get_context_retriever_chain(vector_store, OPENAI_API_KEY):
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)

    retriever = vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            ("user", "Given the above conversation, generate a search query to look up in order to get information relavant to the conversation.")
        ]
    )

    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

    return retriever_chain

def get_conversational_rag_chain(retriever_chain, OPENAI_API_KEY):
    
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)

    # three input here: context = retrieved context , chat_history , user input (question)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ('user', '{input}')
    ])

    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)

    return create_retrieval_chain(retriever_chain, stuff_documents_chain)


# app config
st.set_page_config(page_title="Chat with websites", page_icon="666")

st.title("Welcome To Sia's Chat Bot!\nHere You Can Ask Questions About Any Webpage!")



# sidebar
with st.sidebar:
    # everything here will be shown on sidebar
    st.header("Enter URL and your openAI API key Below")
    website_url = st.text_input("Website URL")

    OPENAI_API_KEY = st.text_input("Your openAI API key")

valid_api_key = False
# try:
#     temp = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
#     valid_api_key = True
# except:
#     valid_api_key = False

if OPENAI_API_KEY is None or OPENAI_API_KEY=="":
    st.info("Please enter your openAI API key")
# elif not valid_api_key:
#     st.info("Entered openAI API key is not valid!")
else:
    if website_url is None or website_url=="":
        st.info("Please enter a website URL")

    else:
        # session state
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = [
                AIMessage(content=f"Welcome to Sia's chatbot, I am here to answer your questions around {website_url} . Let me know what do you have in mind? \U0001f600")
            ]

        # create conversation chain
        if "vector_store" not in st.session_state:
            try:
                st.session_state.vector_store = get_vectorstore_from_url(website_url, OPENAI_API_KEY)
                valid_api_key = True
            except:
                st.info("Entered openAI API key is not valid!")
                valid_api_key = False

        if valid_api_key:
            retriever_chain = get_context_retriever_chain(st.session_state.vector_store, OPENAI_API_KEY)

            conversation_rag_chain = get_conversational_rag_chain(retriever_chain, OPENAI_API_KEY)

            # user input
            user_query = st.chat_input("Type your message here ..")
            if user_query is not None and user_query!="":
                response = get_response(user_query)

                st.session_state.chat_history.append(HumanMessage(content=user_query))
                st.session_state.chat_history.append(AIMessage(content=response))

                
                ## Testing the app
                # chat_history and input variables are going to be replaced
                # retrieved_documents = retriever_chain.invoke({
                #     "chat_history": st.session_state.chat_history,
                #     "input": user_query
                # })
                # st.write(retrieved_documents)
            
            # conversation
            for message in st.session_state.chat_history:
                if isinstance(message, AIMessage):
                    with st.chat_message("AI"):
                        st.write(message.content)
                if isinstance(message, HumanMessage):
                    with st.chat_message("Human"):
                        st.write(message.content)