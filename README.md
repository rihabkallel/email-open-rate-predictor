# Description
This project predicts the open rate of email subject lines based on these features:
- length
- digit count
- exclamation count
- tone
- style

I used LLM chat GPT to analyse the dataset and get the tone and style beforehand.
You can use it for prediction as well and pass the tone and language style as input.
The LLM analysis part was done manually but you can easily automate it by using open AI APIs.

Check my repos here for usage of LLMs: 
https://github.com/rihabkallel/generative-ai-rag-faiss 
https://github.com/rihabkallel/generative-ai-rag-elasticsearch


# Requirements
- Python: >= 3.9
- Venv or Conda for your python virtual environment.
- NotoSansJP-VF.ttf (In case you have a Japanese dataset, I was using it)


# Create a virtual environment
```
python3 -m venv open-rate
source open-rate/bin/activate
```

# Installation
- Install the requirements
```
pip3 install -r requirements.txt
```

# Execution
- Run:
```  python app/main.py ```
- If you want to verify if the embeddings and similarity search work, uncomment line #67 in main.py.
- If you want to execute the LLM generation without RAG, uncomment line #80 in main.py.
- If you want to execute the LLM generation with RAG, use line #89 in main.py.


# More details
