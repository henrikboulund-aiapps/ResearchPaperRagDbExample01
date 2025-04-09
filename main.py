from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.chains import RetrievalQA
import psycopg2


pg_conn = psycopg2.connect(dbname="RagDB", user="postgres", password="mysecretpassword", host="localhost")
pg_cursor = pg_conn.cursor()
pg_cursor.execute("SELECT title, authors, abstract FROM research_papers")
pg_docs = [Document(page_content=row[1], metadata={"source": "postgresql", "id": row[0], "title": row[1], "abstract": row[2]}) for row in pg_cursor.fetchall()]

embedding_model = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = FAISS.from_documents(pg_docs, embedding_model)

vectorstore.save_local("db_faiss_index_research")

vectorstore = FAISS.load_local("db_faiss_index_research", embedding_model, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever()

qa = RetrievalQA.from_chain_type(llm=ChatOllama(model="llama3.2"), retriever=retriever)

query = "What is Retrieval-Augmented Generation for QA"
result = qa.invoke(query)
print(result['result'])