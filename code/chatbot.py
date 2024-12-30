from langchain_community.llms import CTransformers
from langchain.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import json


#* Load config
with open('config/model_cof.json', 'r') as file:
    config_data = json.load(file)

chatbot_path = config_data['chatbot_path']['Q5bit']
chatbot_type = config_data['chatbot_type']
chatbot_config = config_data['chatbot_config']
embedding_path = config_data['embedding_path']


#* Define function to load LLM
def load_llm(model_file, model_type, config, callbacks):
    llm = CTransformers(
        model = model_file,
        model_type = model_type,
        config = config,
        callbacks=[StreamingStdOutCallbackHandler()] if callbacks == 1 else [],
        verbose=False
    )
    return llm

#* Define function to create prompt
def create_prompt(template):
    prompt = PromptTemplate(template = template, input_variables=["question"])
    return prompt

# template = """
# [INST] Bạn là một Chatbot AI để trả lời các câu hỏi về các kiến thức khoa học, thế giới và văn hóa. [/INST]
# Người dùng: {question}
# Chatbot:"""

template = """Bạn là một Chatbot AI để trả lời các câu hỏi về các kiến thức khoa học, thế giới và văn hóa. Người dùng: {question}. Hãy trả lời ngắn gọn và rõ ràng nhất có thể. Nếu không biết câu trả lời, hãy nói không biết, đừng cố tạo ra câu trả lời. Chatbot:"""

prompt = create_prompt(template)

#* Load LLM
llm = load_llm(chatbot_path, chatbot_type, chatbot_config, 1)
llm_chain = prompt | llm

#* Start chatbot
print("Program started, enter your questions.")
while True:
    question = input("Nhập câu hỏi của bạn: ")
        
    if question.lower() == "thoát":
        print("Kết thúc trò chuyện.")
        break

    # Tạo câu trả lời
    llm_chain.invoke({"question": question})
    print("\n\n=============\n")