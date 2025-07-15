# Medical_chatbot_langchain

# How to run?

### STEP 01- Create a conda environment after opening the repository

```bash
python -m venv env
```

```bash
env\Scripts\activate
```


### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```


### Create a `.env` file in the root directory and add your Pinecone & openai credentials as follows:

```ini
PINECONE_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
GROQ_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```


```bash
# run the following command to store embeddings to pinecone
python store_index.py
```

```bash
# Finally run the following command
streamlit run main.py
```



### Techstack Used:

- Python
- LangChain
- streamlit
- Pinecone




	

  
