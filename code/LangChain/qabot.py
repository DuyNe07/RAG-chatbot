from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS
import torch
import re
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 

# Cấu hình
model_file = "models/vinallama-7b-chat_q5_0.gguf"
vector_db_path = "vectorstores/db_faiss"

# Load LLM với max_new_tokens = 4096
def load_llm(model_file):
    llm = CTransformers(
        model=model_file,
        model_type="llama",  # Loại mô hình của bạn (ví dụ llama, gpt-2,...)
        max_new_tokens=4096,  # Giới hạn số token tối đa trong câu trả lời
        temperature=0.5,      # Độ ngẫu nhiên trong câu trả lời
        context_length=2048,  # Độ dài bối cảnh, điều chỉnh nếu cần
        gpu_layers=-1,        # Lớp GPU (để tận dụng GPU)
        device_map="auto",    # Tự động phân bổ các layer trên GPU
        use_gpu=True,         # Sử dụng GPU nếu có
        top_p=0.8,            # Tăng độ đa dạng câu trả lời
        repetition_penalty=1.2, # Trừng phạt sự lặp lại trong câu trả lời
    )
    return llm

# Tạo prompt template
def create_prompt(template):
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    return prompt

# Tạo chain QA
def create_qa_chain(prompt, llm, db):
    llm_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 3}, max_tokens_limit=300),
        return_source_documents=False,
        chain_type_kwargs={'prompt': prompt}
    )
    return llm_chain

# Đọc VectorDB
def read_vectors_db():
    # Embedding
    embedding_model = GPT4AllEmbeddings(model_file="models/all-MiniLM-L6-v2-f16.gguf")
    db = FAISS.load_local(vector_db_path, embedding_model, allow_dangerous_deserialization=True)
    return db

# Khởi động QA
db = read_vectors_db()
llm = load_llm(model_file)

# Tạo Prompt
template = """<|im_start|>system\nUse information from the article to answer the question that addresses the issue in the question. If so, create an answer, if not, don't. Only get the additional result once. Answer the average length and answer in English and complete the idea without leaving it unfinished.\n
    {context}\n<|im_start|>user {question}<|im_end|>\n<|im_start|>assistant """
prompt = create_prompt()

llm_chain = create_qa_chain(prompt, llm, db)

def clean_response(response):
    # Loại bỏ các thẻ <|im_start|>, <|im_end|>, user, assistant, system
    response = re.sub(r"<\|im_start\|>user.*?<\|im_end\|>", '', response)
    response = re.sub(r"<\|im_start\|>assistant", '', response)
    response = re.sub(r"<\|im_end|>", '', response)
    # response = re.sub(r"<|im_start|>.*?<|im_end|>", '', response)  # Xóa tất cả các thẻ
    response = re.sub(r"\|", '', response)  # Xóa tất cả các thẻ
    response = re.sub(r"(system)", '', response)  # Loại bỏ user, assistant, system
    response = re.sub(r'\n+', '. ', response)  # Thay thế tất cả \n bằng dấu chấm
    response = response.strip()  # Loại bỏ khoảng trắng thừa

    return response

def main():
    """Chạy vòng lặp nhập câu hỏi và trả lời cho người dùng."""
    while True:
        question = input("Enter your question: ")
        
        if question.lower() == "exit":
            print("End program.")
            break

        # Truyền vào câu hỏi và context để lấy kết quả
        response = llm_chain.invoke({"query": question})

        # Gộp kết quả và trả về
        answer = response.get('result', 'No have answer')

        final_answer = clean_response(answer)

        # In câu trả lời
        print("Answer: ", final_answer, "\n=============\n")
        print("Answer - true: ", answer, "\n=============\n")
        

if __name__ == "__main__":
    main()
