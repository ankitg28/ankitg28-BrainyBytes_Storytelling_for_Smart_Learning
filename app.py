import os
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

# Access the OpenAI API key
openai_api_key = os.getenv('OPENAI_API_KEY')

# Initialize the OpenAI chat model
chatllm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0.6, model='gpt-3.5-turbo')

# Define the system message
system_message_content = """
You are an educational assistant. Your job is to help undergraduate students understand concepts and definitions by explaining them through short, memorable stories that even a 10-year-old can understand. 
The user will input a topic or a phrase up to 20 words. Based on this input, first give a short definition and then generate a simple story to explain the concept clearly and memorably.
If a user asks a question unrelated to education, kindly notify them that the chatbot is designed for educational purposes only.
"""

# Contact information
linkedin_url = "https://www.linkedin.com/feed/?trk=nav_back_to_linkedin"
email = "ankit.28.goyal@gmail.com"
github_url = "https://github.com/ankitg28"

def main():
    st.set_page_config(page_title="BrainyBytes: Storytelling for Smart Learning", page_icon=":books:")
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
            .contact {{
                display: flex;
                justify-content: center;
                gap: 20px;
                margin-top: 10px;
            }}
            .contact a {{
                text-decoration: none;
                color: white;
                font-weight: bold;
            }}
            .content {{
                padding: 20px;
                background: #f1f1f1;
                border-radius: 10px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
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
        <div class="content">
            <h3>Why I Created BrainyBytes</h3>
            <p>Hi, I'm Ankit! I have always had a problem remembering things. Over time, I discovered that creating a story around a concept helps me remember it much better. This inspired me to create BrainBytes, a small tool to aid in learning by transforming complex concepts into simple, memorable stories. I hope this helps you as much as it helps me!</p>
        </div>
    """, unsafe_allow_html=True)
    st.write("### Ask me about any concept, and I'll explain it with a short story!")
   
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "user_input" not in st.session_state:
        st.session_state.user_input = ""

    # Display the conversation history
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
        "Enter your question (up to 150 characters):", 
        value=st.session_state.user_input, 
        max_chars=150, 
        key="user_input", 
        placeholder="Type your question here..."
    )

    if st.button("Ask Me?"):
        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})

            # Prepare messages
            messages = [
                SystemMessage(content=system_message_content),
                HumanMessage(content=user_input)
            ]

            response = chatllm.invoke(messages)
            assistant_response = response.content

            # Append the assistant's response to the conversation history
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})

            # Clear the input box after sending the message
            st.experimental_rerun()
    
    st.markdown("</div>", unsafe_allow_html=True)

    # Footer section
    st.markdown("""
        <div class="footer">
            <p>Developed by Ankit Goyal</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
