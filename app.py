import os
import re
import pandas as pd
import numpy as np
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer
from groq import Groq

# ========== Configure Groq ==========
os.environ["GROQ_API_KEY"] = "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"
client = Groq(api_key=os.environ["GROQ_API_KEY"])

# A threshold for when the user query should be summarized before subquery generation
QUERY_LENGTH_THRESHOLD = 300  # Feel free to adjust as needed


# ========== Summarization for Large Queries ==========
def summarize_query(long_query: str) -> str:
    """
    Summarizes a long query while preserving main requirement, technical requirements, 
    and skill sets. Uses the Groq LLM for summarization.
    """
    model_name = "llama3-70b-8192"  # or "gemma-7b-it"
    
    prompt = f"""
You are a specialized query summarizer.

The user provided a very long query:
\"\"\"{long_query}\"\"\"

Please write a concise summary that retains:
- The main requirement(s)
- Technical requirements
- Skill sets
- tech stack

Do not omit any critical details. 
The summary must be significantly shorter than the original, 
but still capture all relevant info for generating subqueries and it should no lag in any kind of important information fro the main querie.
"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=512
    )
    summarized_text = response.choices[0].message.content.strip()
    return summarized_text


# ========== LLM Subquery Generator ==========
def generate_subqueries_groq(user_query: str, num_subqueries: int = 5):
    """
    Generates subqueries using Groq LLM to aid in similarity search.
    Specifically preserves main requirements, technical details,
    and skill sets from the input query or summary.
    """
    model_name = "llama3-70b-8192"  # or "gemma-7b-it"
    
    prompt = f"""
You are a high level subquery generator for a semantic search system so u need to keep the important key words of the Querie.

We have the following user query (or query summary):
\"\"\"{user_query}\"\"\"

We need to generate {num_subqueries} concise subqueries 
that ensure we capture all essential aspects from the query and should not keep the irrelevent part fro the querie in the sub queries just keep the part that is required.
(main requirement, technical requirements, skill sets,tech stack, etc.).

Write each subquery on its own line.
"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=512
    )

    generated_text = response.choices[0].message.content.strip()
    subqueries = [line.strip() for line in generated_text.split("\n") if line.strip()]
    return subqueries


# ========== FAISS Retriever Class ==========
class JobSolutionRetriever:
    """
    - Loads a CSV of job solutions (Final_data_with_details.csv).
    - Converts to JSON (list of dicts).
    - Builds a FAISS index by combining multiple columns into a single text for embedding.
    - Allows searching by query.
    """

    def __init__(self, csv_path: str, embedding_model_name="all-MiniLM-L6-v2"):
        """
        :param csv_path: Path to the CSV file containing job solutions.
        :param embedding_model_name: Name of the SentenceTransformer model to use.
        """
        self.csv_path = csv_path
        self.df = None
        self.data_json = []
        self.embeddings = None
        self.index = None
        self.embedding_model = SentenceTransformer(embedding_model_name)

        # 1. Load data
        self.load_data()
        # 2. Convert to JSON
        self.convert_to_json()
        # 3. Build the FAISS vector index
        self.build_vector_index()

    def load_data(self):
        """Load the CSV data into a pandas DataFrame."""
        self.df = pd.read_csv(self.csv_path)

    def convert_to_json(self):
        """Convert the CSV rows to a list of dictionaries."""
        self.data_json = self.df.to_dict(orient='records')

    def build_vector_index(self):
        """
        Build a FAISS index from the combined text of each row.
        Assuming columns like:
          - Solution Type, Solution Name, URL, Remote Testing Support,
            Adaptive/IRT Support, Description, Job Level, Languages, Assessment Length
        """
        combined_texts = []
        for row in self.data_json:
            text_for_embedding = " | ".join([
                str(row.get("Solution Type", "")),
                str(row.get("Solution Name", "")),
                str(row.get("URL", "")),
                str(row.get("Remote Testing Support", "")),
                str(row.get("Adaptive/IRT Support", "")),
                str(row.get("Description", "")),
                str(row.get("Job Level", "")),
                str(row.get("Languages", "")),
                str(row.get("Assessment Length", "")),
            ])
            combined_texts.append(text_for_embedding)

        vecs = self.embedding_model.encode(combined_texts, show_progress_bar=False)
        self.embeddings = np.array(vecs).astype('float32')

        # Create a FAISS index (FlatL2)
        embed_dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(embed_dim)

        # Add embeddings to FAISS index
        self.index.add(self.embeddings)

    def search(self, query: str, top_k: int = 3):
        """
        Embed the user query, then search in FAISS for the top_k results.
        Returns a list of (distance, row_dict).
        """
        query_embedding = self.embedding_model.encode([query]).astype('float32')
        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            results.append((dist, self.data_json[idx]))
        return results


def retrieve_top_solutions(main_query: str, 
                           retriever: JobSolutionRetriever, 
                           num_subqueries: int = 5, 
                           top_k_per_subquery: int = 2, 
                           final_top_k: int = 5):
    """
    1. If the query is large, summarize it.
    2. Generate subqueries using Groq (with the summarized or original query).
    3. For each subquery, retrieve top_k_per_subquery results from FAISS.
    4. Merge results, sort by distance, and return the top final_top_k.
    """

    # 1. Summarize the query if it's too long
    if len(main_query) > QUERY_LENGTH_THRESHOLD:
        summarized_query = summarize_query(main_query)
        # 2. Generate subqueries from the summarized query
        subqueries = generate_subqueries_groq(summarized_query, num_subqueries)
    else:
        # Directly generate subqueries from the original query
        subqueries = generate_subqueries_groq(main_query, num_subqueries)

    all_results = []
    for sq in subqueries:
        sq_results = retriever.search(sq, top_k=top_k_per_subquery)
        all_results.extend(sq_results)

    # Sort all results by ascending distance
    all_results.sort(key=lambda x: x[0])

    final_results = all_results[:final_top_k]

    # Format output
    output = []
    for dist, row_data in final_results:
        solution_name = row_data.get("Solution Name", "")
        description = row_data.get("Description", "")
        remote = row_data.get("Remote Testing Support", "")
        time_duration = row_data.get("Assessment Length", "")
        url = row_data.get("URL", "")

        output.append({
            "title": solution_name,
            "description": description,
            "remote": remote,
            "time_duration": time_duration,
            "url": url
        })

    return output


# ========== STREAMLIT APP ==========

st.title("Recommendation System")

# Hardcode the CSV path in the "backend"
csv_path = r"Final_data_with_details.csv"

# Check if file exists
if not os.path.exists(csv_path):
    st.error(f"CSV not found at path: {csv_path}\n"
             f"Please place 'Final_data_with_details.csv' in the same folder.")
    st.stop()

# Initialize the retriever in session state (only once)
if "retriever" not in st.session_state:
    st.session_state.retriever = JobSolutionRetriever(csv_path)

# Provide input fields
main_query = st.text_input("Enter your query", "")
num_subqueries = st.number_input("Number of Subqueries", min_value=1, value=3, step=1)
top_k_per_subquery = st.number_input("Top-K per subquery", min_value=1, value=2, step=1)
final_top_k = st.number_input("Final top-K", min_value=1, value=5, step=1)

# On "Search" click, generate results
if st.button("Search"):
    results = retrieve_top_solutions(
        main_query=main_query,
        retriever=st.session_state.retriever,
        num_subqueries=num_subqueries,
        top_k_per_subquery=top_k_per_subquery,
        final_top_k=final_top_k
    )

    st.subheader("Results")
    if results:
        for i, item in enumerate(results, start=1):
            st.markdown(f"**[{i}] Title**: {item['title']}")
            st.markdown(f"- **Description**: {item['description']}")
            st.markdown(f"- **Remote**: {item['remote']}")
            st.markdown(f"- **Time Duration**: {item['time_duration']}")
            st.markdown(f"- **URL**: {item['url']}")
            st.markdown("---")
    else:
        st.write("No results found.")
