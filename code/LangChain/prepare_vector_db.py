from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import GPT4AllEmbeddings
import faiss

pdf_data_path = "data/pdf"
vector_db_path = "vectorstores/db_faiss"

# Kiểm tra xem GPU có sẵn không
import torch
if torch.cuda.is_available():
    print(f"GPU available: {torch.cuda.get_device_name(0)}")
else:
    print("GPU không khả dụng, sử dụng CPU")

def create_db_from_text():
    raw_text = """Từ cuối ngày 28.11, giá vàng miếng SJC đã giảm 100.000 đồng/lượng và sáng nay (29.11) đứng yên. Cụ thể, Công ty vàng bạc đá quý Sài Gòn - SJC mua vào 82,9 triệu đồng, bán ra 85,4 triệu đồng; Công ty CP vàng bạc đá quý Phú Nhuận (PNJ) mua vào 82,9 triệu đồng, bán ra 85,4 triệu đồng… Tương tự, vàng nhẫn cũng giảm 100.000 đồng từ cuối ngày hôm qua và không thay đổi trong đầu ngày hôm nay. SJC mua vào 82,5 triệu đồng, bán ra 84,4 triệu đồng; Công ty PNJ mua vào 83,4 triệu đồng, bán ra 84,5 triệu đồng; Công ty Phú Quý mua vào 82,6 triệu đồng, bán ra 84,6 triệu đồng; Doji mua vào với giá 83,5 triệu đồng và bán ra 84,7 triệu đồng… Như vậy, tại một số cửa hàng như PNJ, Doji giá mua vàng nhẫn cao hơn mua vàng miếng từ 500.000 - 600.000 đồng/lượng."""

    # Chia nhỏ văn bản
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=512,
        chunk_overlap=50,
        length_function=len
    )

    chunks = text_splitter.split_text(raw_text)

    # Embedding văn bản
    embedding_model = GPT4AllEmbeddings(model_name="vilm/vinallama-7b-chat", use_gpu=True)  # Chú ý tham số use_gpu=True

    # Đưa vào FAISS VectorDB (chắc chắn FAISS sử dụng GPU)
    db = FAISS.from_texts(texts=chunks, embedding=embedding_model)  # Gọi đúng phương thức embed_documents
    db.save_local(vector_db_path)

    return db
    


def create_db_from_pdf():
    # Load dữ liệu từ PDF
    loader = DirectoryLoader(pdf_data_path, glob="*.pdf", loader_cls=PyMuPDFLoader)
    documents = loader.load()

    # Chia nhỏ văn bản
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=20)
    chunks = text_splitter.split_documents(documents)

    # Embedding văn bản
    embedding_model = GPT4AllEmbeddings(model_file="models/all-MiniLM-L6-v2-f16.gguf", use_gpu=True)  # Đảm bảo sử dụng GPU

    # Tạo FAISS với GPU
    db = FAISS.from_documents(chunks, embedding_model)

    # FAISS sẽ tự động sử dụng GPU nếu bạn đã cài đặt faiss-gpu
    db.save_local(vector_db_path)
    return db


# Gọi hàm tạo DB từ PDF
create_db_from_pdf()
