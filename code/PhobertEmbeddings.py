from transformers import AutoTokenizer, AutoModel
# from langchain.embeddings import Embeddings
import torch
import numpy as np

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

#* 1. Tải mô hình và tokenizer từ HunggingFace
model_name = "vinai/phobert-base-v2"
phobert = AutoModel.from_pretrained(model_name, ignore_mismatched_sizes=True).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

#* 2. Tạo class PhoBertEmbeddings có cấu trúc giống với langchain_community.embeddings
class PhoBERTEmbeddings:
    def __init__(self, model_name="vinai/phobert-base-v2"):
        # Tải mô hình và tokenizer từ Hugging Face
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def embed_documents(self, texts: list) -> list:
        # Chuyển các đoạn văn bản thành embedding
        embeddings = []
        for text in texts:
            # Tokenize văn bản
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            inputs = {key: val.to(device) for key, val in inputs.items()}  # Đảm bảo chuyển inputs lên GPU nếu có
            with torch.no_grad():
                # Lấy embedding từ mô hình
                outputs = self.model(**inputs)
                # Sử dụng lớp [CLS] token để đại diện cho câu
                sentence_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()  # Chuyển kết quả về numpy
                embeddings.append(sentence_embedding)
        return embeddings

    def embed_query(self, text: str) -> np.ndarray:
        # Hàm lấy embedding cho một câu truy vấn
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {key: val.to(device) for key, val in inputs.items()}  # Đảm bảo chuyển inputs lên GPU nếu có
        with torch.no_grad():
            outputs = self.model(**inputs)
            query_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()  # Chuyển kết quả về numpy
        return query_embedding