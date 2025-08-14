from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import gradio as gr

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
chat_history = [
    SystemMessage(content= "You are a helpful ai")
]

chat_active = True

def chat_ai(user_input, history):
    global chat_active
    if user_input.lower() == "exit" or not chat_active:
        chat_active = False
        history.append(["System", "chatbot stopped. refresh to start again."])
        return history,""

    chat_history.append(HumanMessage(content=user_input))
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    history.append([user_input, result.content])
    return history, ""

def stop_chat(history):
    global chat_active
    chat_active = False
    history.append(["System", "Chatbot is stopped. Refresh the page to start again."])
    return history

with gr.Blocks() as demo:
    gr.Markdown("AI Chatbot")
    chatbot_ui = gr.Chatbot()
    msg_box = gr.Textbox(placeholder="Type your message here...")
    with gr.Row():
        send_btn = gr.Button("Send")
        exit_btn = gr.Button("Exit Chat")

    send_btn.click(chat_ai, [msg_box, chatbot_ui], [chatbot_ui, msg_box])
    exit_btn.click(stop_chat, [chatbot_ui], [chatbot_ui])

demo.launch()

# while True:
#     user_input = input('You: ')
#     chat_history.append(HumanMessage(content=user_input))
#     if user_input == 'exit':
#         break
#     result = model.invoke(chat_history)
#     chat_history.append(AIMessage(content=result.content))
#     print("AI:", result.content)

# print(chat_history)
