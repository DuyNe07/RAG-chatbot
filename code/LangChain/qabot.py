from langchain_community.llms import GPT4All, CTransformers
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS
import faiss
# Cấu hình
model_file = "models/vinallama-7b-chat_q5_0.gguf"
vector_db_path = "vectorstores/db_faiss"

# Load LLM với max_new_tokens = 4096
# def load_llm(model_file):
#     llm = CTransformers(
#         model=model_file,
#         model_type="llama",  # Loại mô hình của bạn (ví dụ llama, gpt-2,...)
#         max_new_tokens=4096,  # Giới hạn số token tối đa trong câu trả lời
#         temperature=0.5,      # Độ ngẫu nhiên trong câu trả lời
#         context_length=2048,  # Độ dài bối cảnh, điều chỉnh nếu cần
#         gpu_layers=-1,        # Lớp GPU (để tận dụng GPU)
#         top_p=0.8,            # Tăng độ đa dạng câu trả lời
#         repetition_penalty=1.2, # Trừng phạt sự lặp lại trong câu trả lời
#     )
#     return llm

def load_llm(model_file):
    llm = GPT4All(
        model=model_file,  # Đường dẫn đến mô hình của bạn (ví dụ: "models/vinallama-7b-chat_q5_0.gguf")
        max_tokens=2048,             # Độ dài bối cảnh (context length)
        temp=0.5,        # Độ ngẫu nhiên trong câu trả lời
        top_p=0.8,              # Tăng độ đa dạng câu trả lời
        repeat_penalty=1.2,     # Trừng phạt sự lặp lại trong câu trả lời
        device='gpu',           # Sử dụng GPU nếu có
        n_threads=8          
    )
    return llm

# Tạo prompt template (chỉ sử dụng question)
def create_prompt(template):
    prompt = PromptTemplate(template = template, input_variables=["context", "question"])
    return prompt

# Tạo chain QA
def create_qa_chain(prompt, llm, db):
    llm_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # Loại chain cho việc trả lời câu hỏi
        retriever=db.as_retriever(search_kwargs={"k": 3}, max_tokens_limit=300),
        return_source_documents=False,
        chain_type_kwargs={'prompt': prompt}  # Truyền prompt vào cho chain
    )
    return llm_chain

def create_llm_chain(prompt, llm):
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    return llm_chain

# Đọc VectorDB
def read_vectors_db():
    # Embedding
    embedding_model = GPT4AllEmbeddings(model_file="models/vinallama-7b-chat_q5_0.gguf")

    # Tải DB FAISS từ ổ cứng
    db = FAISS.load_local(vector_db_path, embedding_model, allow_dangerous_deserialization=True)

    # Nếu có GPU, chuyển FAISS index lên GPU
    if faiss.get_num_gpus() > 0:
        print("Using GPU for FAISS")
        # Chuyển FAISS index từ CPU lên GPU
        res = faiss.StandardGpuResources()  # Tạo tài nguyên GPU
        db.index = faiss.index_cpu_to_gpu(res, 0, db.index)  # Chuyển chỉ mục lên GPU (sử dụng GPU 0)
    else:
        print("No GPU found, using CPU for FAISS")

    return db

# Khởi động QA
db = read_vectors_db()
llm = load_llm(model_file)

# Tạo Prompt (chỉ sử dụng question)
template = """<|im_start|>system\nSử dụng thông tin sau đây để trả lời câu hỏi. Nếu bạn không biết câu trả lời, hãy nói không biết, đừng cố tạo ra câu trả lời\n
    {context}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant"""
prompt = create_prompt(template)

# Tạo llm_chain
llm_chain = create_qa_chain(prompt, llm, db)
# llm_chain = create_llm_chain(prompt, llm)

def main():
    """Chạy vòng lặp nhập câu hỏi và trả lời cho người dùng."""
    while True:
        question = input("Enter your question: ")
        
        if question.lower() == "exit":
            print("End program.")
            break

        context = "Chúng ta đang nói chuyện các vấn đề liên quan đến thông tin toán học"  # Cung cấp context nếu có
        response = llm_chain.run({"query": question})  # Truyền vào context và câu hỏi

        # QA pdf
        # response = llm_chain.invoke({"query": question})  # Chỉ truyền vào câu hỏi

        # Gộp kết quả và trả về
        answer = response.get('result', 'No answer found')

        print("Answer: ", answer, "\n=============\n")

if __name__ == "__main__":
    main()