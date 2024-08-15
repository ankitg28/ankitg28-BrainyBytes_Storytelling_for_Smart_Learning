import os
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import HumanMessage, SystemMessage
from dotenv import load_dotenv
import openai
from langchain_community.document_loaders import Docx2txtLoader
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from openai import OpenAI
# Load environment variables
load_dotenv()

# Contact information
linkedin_url = "https://www.linkedin.com/in/goyalankit28/"
email = "ankit.28.goyal@gmail.com"
github_url = "https://github.com/ankitg28"

# Access the OpenAI API key
openai_api_key = os.getenv('OPENAI_API_KEY')
openai.api_key = openai_api_key
client=OpenAI()
# Initialize the OpenAI chat model for story generation
chatllm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0.6, model=os.getenv('MODEL_NAME'))

# System message for the assistant
system_message_content = """
You are an educational assistant. Your job is to help undergraduate students understand concepts and definitions by explaining them through short, memorable stories that even a 10-year-old can understand. 
The user will input a topic or a phrase up to 20 words. Based on this input, first give a short definition and then generate a simple story to explain the concept clearly and memorably.
If a user asks a question unrelated to education, kindly notify them that the chatbot is designed for educational purposes only.
"""

# Function to load the PDF into a vector database
def load_db(file_path, file_type):
    if file_type == "pdf":
        loader = PyPDFLoader(file_path)
    elif file_type == "docx":
        loader = Docx2txtLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectordb = Chroma.from_documents(docs, embeddings)
    retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 3})
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=chatllm,
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain

# Function to get the definition from the RAG pipeline
def get_definition(query, qa_chain):
    result = qa_chain({
        "question": query,
        "chat_history": []
    })
    return result["answer"]

# Function to generate the story based on the retrieved definition
def generate_story(definition, story_length):
    story_prompt = f"Create a simple and memorable story for a 10-year-old that explains: {definition}. Please keep the story within {story_length} words."
    messages = [
        SystemMessage(content=system_message_content),
        HumanMessage(content=story_prompt)
    ]
    response = chatllm.invoke(messages)
    story_content = response.content
    if "Definition:" in story_content:
        story_content = story_content.split("Definition:")[-1].strip()
    if "Story:" in story_content:
        story_content = story_content.split("Story:")[-1].strip()
    return story_content

# Function to generate a DALLE prompt based on the definition and story
def generate_dalle_prompt(definition, story):
    prompt = f"An illustration that visually represents the concept: {definition}. Story details: {story}"
    return prompt

# Function to generate the DALLE image using OpenAI's API
def genImage(input_prompt):
    try:
        response = client.images.generate(
            prompt=input_prompt,
            n=1,
            size="1024x1024"
        )
        return response['data'][0]['url']
    except Exception as e:
        st.error(f"Error generating DALLE image: {e}")
        return None

# Function to generate quiz questions based on the definition
def generate_quiz(definition):
    quiz_prompt = f"Based on the following definition, generate three quiz questions to test the user's understanding of the topic:\n\nDefinition: {definition}\n\nProvide the questions in a structured format with multiple-choice options."
    messages = [
        SystemMessage(content="You are a quiz generator. Generate quiz questions to test a student's understanding of the topic."),
        HumanMessage(content=quiz_prompt)
    ]
    response = chatllm.invoke(messages)
    return response.content

def main():
    st.set_page_config(page_title="BrainBytes: Storytelling for Smart Learning", page_icon=":books:")

    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None

    if "selected_tab" not in st.session_state:
        st.session_state.selected_tab = "Chatbot"

    # Dropdown for selecting the section instead of radio buttons
    menu = st.sidebar.selectbox("Choose a section:", ["Chatbot","Upload File", "View Data", ], index=[ "Chatbot","Upload File", "View Data"].index(st.session_state.selected_tab))

    st.session_state.selected_tab = menu

    # Apply consistent styling to the "Upload File" tab
    if menu == "Upload File":
        st.markdown(f"""
            <style>
                .header {{
                    padding: 20px;
                    text-align: center;
                    background: linear-gradient(to right, #a1c4fd, #c2e9fb);
                    border-radius: 10px;
                    margin-bottom: 20px;
                    color: white;
                }}
                .upload-section {{
                    padding: 20px;
                    background: white;
                    border-radius: 10px;
                    box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
                }}
                .footer {{
                    padding: 10px;
                    text-align: center;
                    background: #f1f1f1;
                    border-radius: 10px;
                    margin-top: 20px;
                }}
                .footer p {{
                    margin: 0;
                    color: #6a11cb;
                }}
            </style>
            <div class="header">
                <h1>BrainyBytes: Storytelling for Smart Learning</h1>
                <p>Helping you understand complex concepts with simple stories</p>
                <div class="contact">
                    <a href="{linkedin_url}" target="_blank">LinkedIn</a>
                    <a href="mailto:{email}">Email</a>
                    <a href="{github_url}" target="_blank">GitHub</a>
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.write("### Upload a file (PDF or Word) to load the data for interactive Q&A")

        with st.container():
            uploaded_file = st.file_uploader("Choose a PDF or Word file", type=["pdf", "docx"])
            if uploaded_file:
                file_type = uploaded_file.name.split(".")[-1].lower()
                if file_type == "pdf":
                    with open("temp.pdf", "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    st.session_state.qa_chain = load_db("temp.pdf", "pdf")
                    st.success("PDF successfully loaded into the vector database!")
                elif file_type == "docx":
                    with open("temp.docx", "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    st.session_state.qa_chain = load_db("temp.docx", "docx")
                    st.success("Word document successfully loaded into the vector database!")
                else:
                    st.error("Unsupported file type. Please upload a PDF or Word file.")

    # Keep the rest of your existing tabs (View Data and Chatbot) consistent
    # Code for the "View Data" and "Chatbot" tabs would go here...

        st.markdown("""
            <div class="footer">
                <p>Developed by Ankit Goyal</p>
            </div>
        """, unsafe_allow_html=True)

    elif menu == "View Data":
        st.markdown(f"""
            <style>
                .header {{
                    padding: 20px;
                    text-align: center;
                    background: linear-gradient(to right, #a1c4fd, #c2e9fb);
                    border-radius: 10px;
                    margin-bottom: 20px;
                    color: white;
                }}
                .message {{
                    background: white;
                    border-radius: 10px;
                    padding: 10px;
                    margin-bottom: 10px;
                }}
                .message.vector {{
                    border-left: 5px solid #fd7e14;
                }}
                .vector-content {{
                    color: #6a11cb;
                    font-weight: bold;
                }}
                .vector-metadata {{
                    color: #2575fc;
                    margin-top: 10px;
                }}
                .footer {{
                    padding: 10px;
                    text-align: center;
                    background: #f1f1f1;
                    border-radius: 10px;
                    margin-top: 20px;
                }}
                .footer p {{
                    margin: 0;
                    color: #6a11cb;
                }}
            </style>
            <div class="header">
                <h1>BrainyBytes: Storytelling for Smart Learning</h1>
                <p>Helping you understand complex concepts with simple stories</p>
                <div class="contact">
                    <a href="{linkedin_url}" target="_blank">LinkedIn</a>
                    <a href="mailto:{email}">Email</a>
                    <a href="{github_url}" target="_blank">GitHub</a>
                </div>
            </div>
        """, unsafe_allow_html=True)

        if "qa_chain" in st.session_state:
            if st.button("View Vectors"):
                # Access the vector database from the retriever
                vectordb = st.session_state.qa_chain.retriever.vectorstore
                
                # Access the internal collection
                collection = vectordb._collection

                # Retrieve documents and metadata (embeddings are stored separately)
                stored_documents = collection.get()['documents'][:5]  # Get the first 5 documents
                stored_metadata = collection.get()['metadatas'][:5]   # Get metadata for those documents

                # Display each document's content and metadata
                for i, (doc, meta) in enumerate(zip(stored_documents, stored_metadata)):
                    st.markdown(f"""
                        <div class='message vector'>
                            <div class='vector-content'>**Document {i + 1} Content:**</div>
                            <div>{doc}</div>
                            <div class='vector-metadata' style='margin-top: 10px;'>**Metadata:** {meta}</div>
                        </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("No PDF loaded yet. Please upload a PDF in the 'Upload File' section.")

        st.markdown("""
            <div class="footer">
                <p>Developed by Ankit Goyal</p>
            </div>
        """, unsafe_allow_html=True)


    elif menu == "Chatbot":
        st.markdown(f"""
            <style>
                .header {{
                    padding: 20px;
                    text-align: center;
                    background: linear-gradient(to right, #a1c4fd, #c2e9fb);
                    border-radius: 10px;
                    margin-bottom: 20px;
                    color: white;
                }}
                .message {{
                    background: white;
                    border-radius: 10px;
                    padding: 10px;
                    margin-bottom: 10px;
                }}
                .message.user {{
                    border-left: 5px solid #6a11cb;
                }}
                .message.assistant {{
                    border-left: 5px solid #2575fc;
                }}
                .assistant-definition {{
                    color: #6a11cb;
                    font-weight: bold;
                }}
                .assistant-story {{
                    color: #2575fc;
                    margin-top: 10px;
                }}
                .footer {{
                    padding: 10px;
                    text-align: center;
                    background: #f1f1f1;
                    border-radius: 10px;
                    margin-top: 20px;
                }}
                .footer p {{
                    margin: 0;
                    color: #6a11cb;
                }}
                .content {{
                padding: 20px;
                background: #f1f1f1;
                border-radius: 10px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                }}
                .button-row {{
                    display: flex;
                    justify-content: space-evenly;
                    margin-top: 20px;
                }}
            </style>
            <div class="header">
                <h1>BrainyBytes: Storytelling for Smart Learning</h1>
                <p>Helping you understand complex concepts with simple stories</p>
                <div class="contact">
                    <a href="{linkedin_url}" target="_blank">LinkedIn</a>
                    <a href="mailto:{email}">Email</a>
                    <a href="{github_url}" target="_blank">GitHub</a>
                </div>
            </div>
        """, unsafe_allow_html=True)
        st.write("### Stumped by a tricky concept? 🧐 Let me spin it into a story!")

        if "messages" not in st.session_state:
            st.session_state.messages = []

        if "user_input" not in st.session_state:
            st.session_state.user_input = ""

        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"<div class='message user'><strong>You:</strong> {message['content']}</div>", unsafe_allow_html=True)
            elif message["role"] == "assistant":
                if "Definition:" in message['content'] and "Story:" in message['content']:
                    response_parts = message['content'].split("Story:", 1)
                    definition = response_parts[0].replace("Definition:", "").strip()
                    story = response_parts[1].strip() if len(response_parts) > 1 else ""
                    st.markdown(f"""
                        <div class='message assistant'>
                            <div class='assistant-definition'>Definition: {definition}</div>
                            <div class='assistant-story'>Story: {story}</div>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='message assistant'><strong>Assistant:</strong> {message['content']}</div>", unsafe_allow_html=True)

        user_input = st.text_area(
            "Enter your mind-boggling concept (up to 150 characters):", 
            value=st.session_state.user_input, 
            max_chars=150, 
            key="user_input", 
            placeholder="Go ahead, challenge me with your toughest concept..."
        )

        story_length = st.slider("Select the length of the story (in words):", min_value=50, max_value=300, value=100, step=50)

        # Arrange buttons in a single row
        with st.container():
            col1, col2, col3 = st.columns(3)
            with col1:
                story_button = st.button("Storyify It! 📖")
            with col2:
                image_button = st.button("Generate DALLE Image 🎨")
            with col3:
                quiz_button = st.button("Generate Quiz 📝")

        if story_button:
            if user_input:
                st.session_state.messages.append({"role": "user", "content": user_input})

                if st.session_state.qa_chain:
                    definition = get_definition(user_input, st.session_state.qa_chain)
                else:
                    st.warning("Please upload a PDF in the 'Upload PDF' section.")
                    return

                story = generate_story(definition, story_length)
                assistant_response = f"Definition: {definition}\n\nStory: {story}"

                st.session_state.messages.append({"role": "assistant", "content": assistant_response})

        if image_button:
            if st.session_state.messages:
                last_message = st.session_state.messages[-1]
                if "Definition:" in last_message["content"] and "Story:" in last_message["content"]:
                    definition = last_message["content"].split("Definition:")[1].split("Story:")[0].strip()
                    story = last_message["content"].split("Story:")[1].strip()

                    dalle_prompt = generate_dalle_prompt(definition, story)
                    image_url = genImage(dalle_prompt)

                    if image_url:
                        st.image(image_url, width=600)
                    else:
                        st.error("Failed to generate the DALLE image. Please try again.")

        if quiz_button:
            if st.session_state.messages:
                last_message = st.session_state.messages[-1]
                if "Definition:" in last_message["content"]:
                    definition = last_message["content"].split("Definition:")[1].split("Story:")[0].strip()

                    quiz = generate_quiz(definition)
                    st.markdown(f"""
                        <div class='message assistant'>
                            <div class='assistant-definition'>Quiz Questions:</div>
                            <div class='assistant-quiz' style='margin-top: 10px;'>
                                {quiz}
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

        st.markdown("""
            <div class="footer">
                <p>Developed by Ankit Goyal</p>
            </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
