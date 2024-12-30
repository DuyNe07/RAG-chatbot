from langchain_community.llms import LlamaCpp
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
# from langchain.chains import RunnableSequence

model_file = "models/vinallama-7b-chat_q5_0.gguf"
vector_db_path = "vectorstores/db_faiss"

n_gpu_layers = -1  # The number of layers to put on the GPU. The rest will be on the CPU. If you don't know how many layers there are, you can use -1 to move all to GPU.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

def load_llm(model_file):
    print("Loading model...")
    llm = LlamaCpp(
        model_path=model_file,
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        n_ctx=4096,
        f16_kv=True,
        # callbacks=[StreamingStdOutCallbackHandler()],
        verbose=False,
    )
    return llm
    
def create_prompt(template):
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    return prompt

def create_llm_chain(prompt, llm):
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    return llm_chain

# Load model
llm = load_llm(model_file)

# Tạo Prompt (chỉ sử dụng question và context)
template = """<|im_start|>system\nSử dụng thông tin sau đây để trả lời câu hỏi. Nếu bạn không biết câu trả lời, hãy nói không biết, đừng cố tạo ra câu trả lời<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant"""
prompt = create_prompt(template)

# Tạo llm_chain
llm_chain = prompt | llm

def main():
    """Chạy vòng lặp nhập câu hỏi và trả lời cho người dùng."""
    print("Program started, enter your questions.")
    while True:
        question = input("Enter your question: ")
        
        if question.lower() == "exit":
            print("End program.")
            break

        # Tạo câu trả lời
        response = llm_chain.invoke({"question": question})
        print("Answer:", response)
        print("=============\n")

if __name__ == "__main__":
    main()