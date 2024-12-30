# RAG-chatbot

## Setting environments

1. Install cuda at : [Cuda](https://pytorch.org/get-started/locally/) (specify at CUDA 12.1)
2. Double check  output of script `python -c "import torch; print(torch.cuda.device_count())"` is 1 if you have 1 GPU
3. gpt4all-2.8.2
4. tqdm-4.67.1
5. langchain_community-0.3.9
6. langchain-0.3.9
7. pip install ctransformers[cuda]
8. InstructorEmbedding
9. pymupdf
10. 