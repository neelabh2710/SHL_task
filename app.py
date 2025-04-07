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
QUERY_LENGTH_THRESHOLD = 100  # Adjust as needed

# ========== Summarization for Large Queries ==========
def summarize_query(long_query: str) -> str:
    """
    Summarizes a long query while preserving main requirement, 
    technical requirements, and skill sets. Uses the Groq LLM for summarization.
    """
    model_name = "llama3-70b-8192"  # or "gemma-7b-it"
    
    prompt = f"""
    You are a specialized query summarizer.

    The user provided a lengthy job description or query:
    \"\"\"{long_query}\"\"\" 

    Your task is to write a concise summary that retains:
    - Main requirement(s)
    - Technical requirements
    - Skill sets
    - Tech stack
    - Any other critical information from the original query

    Additionally:
    - Do NOT omit any crucial details related to the job description or expectations.
    - The summary must be significantly shorter than the original.
    - The resulting summary should be highly relevant for generating subqueries
      without losing essential context or requirements.

    Please provide only the summarized text.
    """

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=200
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
    You are a high-level subquery generator for a semantic search system.
    
    The user query (or query summary) is:
    \"\"\"{user_query}\"\"\" 
    
    Your goal is to generate {num_subqueries} concise, precise subqueries that:
    - Retain the main requirement(s)
    - Emphasize critical technical requirements
    - Include relevant skill sets and tech stack
    - Exclude any information that is not essential to the query
    
    Each subquery should focus on a distinct aspect of the user query 
    to ensure comprehensive coverage without duplication.
    
    Write each subquery on its own line. 
    Do not add commentary or extra text beyond these subqueries.
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

    def __init__(self, csv_path: str, embedding_model_name="local_model"):
        """
        :param csv_path: Path to the CSV file containing job solutions.
        :param embedding_model_name: Name of the SentenceTransformer model to use.
                                     Default is "local_model" for your local embedding.
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


# ========== Rerank Helper (Local Model) ==========
def re_rank_with_local_model(all_results, main_query, embedding_model, final_top_k=5):
    """
    Re-rank combined (distance, row_dict) results using the SAME local model,
    but embedding the *main_query* (or its summary) to compute a final similarity.
    """
    # 1. Embed main query (or summarized version if query is large)
    main_query_emb = embedding_model.encode([main_query])[0]

    # 2. For each candidate, embed "Description + Solution Name" for final scoring
    re_ranked = []
    for dist, row_data in all_results:
        text_for_embedding = row_data.get("Description", "") + " " + row_data.get("Solution Name", "")
        doc_emb = embedding_model.encode([text_for_embedding])[0]

        # Cosine similarity
        similarity = np.dot(main_query_emb, doc_emb) / (
            np.linalg.norm(main_query_emb) * np.linalg.norm(doc_emb)
        )

        re_ranked.append((similarity, row_data))

    # 3. Sort by descending similarity
    re_ranked.sort(key=lambda x: x[0], reverse=True)

    # 4. Return top_k
    re_ranked_top = re_ranked[:final_top_k]

    # 5. Format final output
    output = []
    for sim, row_data in re_ranked_top:
        output.append({
            "similarity_score": sim,
            "title": row_data.get("Solution Name", ""),
            "description": row_data.get("Description", ""),
            "remote": row_data.get("Remote Testing Support", ""),
            "time_duration": row_data.get("Assessment Length", ""),
            "url": row_data.get("URL", "")
        })
    return output


# ========== RETRIEVE + RERANK PIPELINE ==========
def retrieve_top_solutions(
    main_query: str, 
    retriever: JobSolutionRetriever, 
    num_subqueries: int = 5, 
    top_k_per_subquery: int = 2, 
    final_top_k: int = 5
):
    """
    1. If the query is large, summarize it.
       - Then use that summary for subquery generation 
         AND for final re-ranking (rather than the original query).
    2. Otherwise, use the original query for both subquery generation & final re-ranking.
    3. For each subquery, retrieve top_k_per_subquery results from FAISS.
    4. Club all subquery results together (optionally remove duplicates).
    5. Re-rank them using the (summarized or original) main query 
       with the same local model.
    6. Return the final top_k results.
    """

    if len(main_query) > QUERY_LENGTH_THRESHOLD:
        # Summarize if the query is large
        summarized_query = summarize_query(main_query)
        # Use summarized_query for subqueries
        subqueries = generate_subqueries_groq(summarized_query, num_subqueries)
        # Also use summarized_query for final ranking
        query_for_rerank = summarized_query
    else:
        # Use the main query as is
        subqueries = generate_subqueries_groq(main_query, num_subqueries)
        query_for_rerank = main_query

    # 2. Gather top_k results from each subquery
    all_results = []
    for sq in subqueries:
        sq_results = retriever.search(sq, top_k=top_k_per_subquery)
        all_results.extend(sq_results)

    # 3. Remove duplicates if necessary (using 'URL' as a unique key)
    unique_map = {}
    for dist_val, row_data in all_results:
        key = row_data.get("URL", "")
        if key not in unique_map:
            unique_map[key] = (dist_val, row_data)
        else:
            # Keep whichever has the lower distance
            if dist_val < unique_map[key][0]:
                unique_map[key] = (dist_val, row_data)

    combined_unique_results = list(unique_map.values())  # => [(dist, row_dict), ...]

    # 4. Re-rank with the local model using query_for_rerank
    final_results = re_rank_with_local_model(
        all_results=combined_unique_results,
        main_query=query_for_rerank,
        embedding_model=retriever.embedding_model,
        final_top_k=final_top_k
    )

    return final_results

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
    st.session_state.retriever = JobSolutionRetriever(
        csv_path=csv_path, 
        embedding_model_name="local_model"
    )

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
            st.markdown(f"- **Similarity Score**: {item['similarity_score']:.4f}")
            st.markdown(f"- **Description**: {item['description']}")
            st.markdown(f"- **Remote**: {item['remote']}")
            st.markdown(f"- **Time Duration**: {item['time_duration']}")
            st.markdown(f"- **URL**: {item['url']}")
            st.markdown("---")
    else:
        st.write("No results found.")
