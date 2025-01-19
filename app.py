import streamlit as st
import os
import re
import nltk
import tiktoken
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from pinecone import Pinecone, ServerlessSpec, Vector
from concurrent.futures import ThreadPoolExecutor
from langchain_community.llms import OpenAI

# Set environment variables
import streamlit as st

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]

if not OPENAI_API_KEY or not PINECONE_API_KEY:
    raise ValueError("API keys for OpenAI and Pinecone are not set in the environment variables.")

# Initialize Pinecone
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index_name = "youtube-transcripts"

# Check or create Pinecone index
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(index_name)

# Chunk transcript into manageable sizes
def chunk_transcript(text, max_tokens=3000, model="gpt-3.5-turbo"):
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    chunks, current_chunk_tokens = [], []

    for token in tokens:
        current_chunk_tokens.append(token)
        if len(current_chunk_tokens) >= max_tokens:
            chunks.append(encoding.decode(current_chunk_tokens))
            current_chunk_tokens = []
    if current_chunk_tokens:
        chunks.append(encoding.decode(current_chunk_tokens))
    return chunks

# Fetch YouTube transcript
def get_youtube_transcript(video_url, chunk_size=4095):
    try:
        video_id_match = re.search(r"v=([a-zA-Z0-9_-]+)", video_url)
        if not video_id_match:
            raise ValueError("Invalid YouTube URL.")
        video_id = video_id_match.group(1)

        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        formatter = TextFormatter()
        text_blob = formatter.format_transcript(transcript)

        chunks = chunk_transcript(text_blob, max_tokens=chunk_size)
        return video_id, chunks
    except Exception as e:
        st.error(f"Error fetching transcript: {e}")
        return None, None
    
# Function to check if chunks already exist in Pinecone
def chunks_exist_in_pinecone(video_id):
    try:
        query_response = index.query(
            vector=[0] * 1536,  # Dummy vector for the query
            top_k=1,            # Check for any matching chunk
            filter={"video_id": {"$eq": video_id}},
            include_metadata=True
        )
        return len(query_response.matches) > 0
    except Exception as e:
        st.error(f"Error checking existing chunks in Pinecone: {e}")
        return False

# Function to store video chunks in Pinecone (only if they don't already exist)
# Function to store video chunks in Pinecone with parallelization
def store_video_in_pinecone(video_id, chunks):
    try:
        if chunks_exist_in_pinecone(video_id):
            st.info(f"Chunks for video ID {video_id} already exist in Pinecone. Skipping storage.")
            return True

        embedding_model = OpenAIEmbeddings(api_key=os.environ["OPENAI_API_KEY"])
        vectors_to_upsert = []

        # Function to process a single chunk
        def process_chunk(i, chunk):
            try:
                embedding = embedding_model.embed_query(chunk)
                text_ref = save_text_externally(video_id, i, chunk)

                if not text_ref:
                    st.warning(f"Failed to save chunk {i}. Skipping...")
                    return None

                metadata = {
                    "video_id": video_id,
                    "chunk_id": i,
                    "chunk_total": len(chunks),
                    "text_ref": text_ref,
                    "is_chunk": True
                }
                return Vector(id=f"{video_id}_{i}", values=embedding, metadata=metadata)
            except Exception as e:
                st.error(f"Error processing chunk {i}: {e}")
                return None

        # Use ThreadPoolExecutor to process chunks in parallel
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(lambda x: process_chunk(*x), enumerate(chunks, start=1)))

        # Collect successful vectors
        vectors_to_upsert = [vector for vector in results if vector is not None]

        # Upsert chunks in parallel batches
        def upsert_batch(batch):
            try:
                index.upsert(vectors=batch)
            except Exception as e:
                st.error(f"Error upserting batch: {e}")

        batch_size = 100
        with ThreadPoolExecutor(max_workers=4) as executor:
            executor.map(upsert_batch, [vectors_to_upsert[i:i + batch_size] for i in range(0, len(vectors_to_upsert), batch_size)])

        st.success(f"Successfully stored {len(vectors_to_upsert)} chunks for video ID {video_id}.")
        return True
    except Exception as e:
        st.error(f"Error storing video {video_id} in Pinecone: {e}")
        return False


# Save transcript chunks externally
def save_text_externally(video_id, chunk_id, text):
    try:
        file_path = f"{video_id}_chunk_{chunk_id}.txt"
        with open(file_path, 'w') as file:
            file.write(text)
        return file_path
    except Exception as e:
        st.error(f"Error saving chunk {chunk_id} to file: {e}")
        return None

# Store chunks in Pinecone
def store_video_in_pinecone(video_id, chunks):
    try:
        embedding_model = OpenAIEmbeddings(api_key=os.environ["OPENAI_API_KEY"])
        vectors_to_upsert = []

        for i, chunk in enumerate(chunks, start=1):
            try:
                embedding = embedding_model.embed_query(chunk)
                text_ref = save_text_externally(video_id, i, chunk)

                if not text_ref:
                    st.warning(f"Failed to save chunk {i}. Skipping...")
                    continue

                metadata = {
                    "video_id": video_id,
                    "chunk_id": i,
                    "chunk_total": len(chunks),
                    "text_ref": text_ref,
                    "is_chunk": True
                }
                vector = Vector(id=f"{video_id}_{i}", values=embedding, metadata=metadata)
                vectors_to_upsert.append(vector)
            except Exception as e:
                st.error(f"Error processing chunk {i}: {e}")

        for i in range(0, len(vectors_to_upsert), 100):
            index.upsert(vectors=vectors_to_upsert[i:i + 100])

        st.success(f"Successfully stored {len(vectors_to_upsert)} chunks for video ID {video_id}.")
        return True
    except Exception as e:
        st.error(f"Error storing video {video_id} in Pinecone: {e}")
        return False

# Query chunks from Pinecone
# Query chunks from Pinecone with parallelization
def query_video_chunks(video_id):
    try:
        query_response = index.query(
            vector=[0] * 1536,
            top_k=1000,
            include_metadata=True,
            filter={"video_id": {"$eq": video_id}}
        )
        matches = query_response.matches

        # Function to process a single match
        def process_match(match):
            chunk_id = match.metadata.get('chunk_id')
            text_ref = match.metadata.get('text_ref')
            if text_ref and os.path.exists(text_ref):
                with open(text_ref, 'r') as file:
                    return chunk_id, file.read().strip()
            return None, None

        # Use ThreadPoolExecutor to process matches in parallel
        with ThreadPoolExecutor(max_workers=8) as executor:
            chunks = list(executor.map(process_match, matches))

        # Collect and sort valid chunks
        chunks = [chunk for chunk in chunks if chunk[1]]
        chunks.sort(key=lambda x: x[0])
        return [chunk[1] for chunk in chunks]
    except Exception as e:
        st.error(f"Error querying video chunks: {e}")
        return None


# Function to refine query with LLM
def refine_query_with_llm(video_id, user_query):
    try:
        chunks = query_video_chunks(video_id)
        if not chunks:
            return "No data available for this video."

        # Initialize OpenAI LLM
        llm = OpenAI(temperature=0.7, max_tokens=500)

        # Token encoding for token calculations
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        max_context_length = 4097  # Model's maximum token limit
        completion_tokens = 500    # Reserved tokens for completion

        # Static prompt structure
        static_prompt_template = """
        Based on the following portion of the transcript, address the task below:
        
        Transcript Portion:
        {chunk}

        Task: {user_query}
        """
        static_prompt_tokens = len(encoding.encode(static_prompt_template.format(chunk="", user_query=user_query)))
        max_chunk_tokens = max_context_length - static_prompt_tokens - completion_tokens

        responses = []

        # Process each chunk
        for i, chunk in enumerate(chunks, start=1):
            try:
                chunk_tokens = encoding.encode(chunk)

                # Dynamic truncation with retry logic
                while len(chunk_tokens) > max_chunk_tokens:
                    st.warning(f"Truncating chunk {i} further to fit within the token limit.")
                    chunk = encoding.decode(chunk_tokens[:max_chunk_tokens])
                    chunk_tokens = encoding.encode(chunk)  # Recalculate token length

                # Construct the prompt
                prompt = static_prompt_template.format(chunk=chunk, user_query=user_query)

                # Final validation of prompt length
                prompt_tokens = len(encoding.encode(prompt))
                if prompt_tokens + completion_tokens > max_context_length:
                    st.error(f"Prompt for chunk {i} still exceeds token limit after truncation: {prompt_tokens} tokens.")
                    continue  # Skip this chunk

                # Invoke the model
                response = llm.invoke(prompt)

                # Collect responses
                if isinstance(response, str):
                    responses.append(response.strip())
                elif hasattr(response, "content"):
                    responses.append(response.content.strip())
                else:
                    st.warning(f"Unexpected response format for chunk {i}. Skipping...")
            except Exception as e:
                st.error(f"Error processing chunk {i}: {e}")

        # Combine responses into the final synthesis prompt
        if not responses:
            return "No valid responses were generated from the chunks."

        final_prompt = f"""
        Using the following responses extracted from the transcript, perform the task below:

        Responses from Transcript:
        {' '.join(responses)}

        Task: {user_query}
        """
        final_prompt_tokens = len(encoding.encode(final_prompt))
        max_final_prompt_tokens = max_context_length - completion_tokens

        # Truncate the final synthesis prompt if needed
        if final_prompt_tokens > max_final_prompt_tokens:
            st.warning("Truncating final synthesis prompt to fit within the token limit.")
            final_prompt = encoding.decode(encoding.encode(final_prompt)[:max_final_prompt_tokens])

        # Invoke the final synthesis step
        final_response = llm.invoke(final_prompt)

        # Return the response
        if isinstance(final_response, str):
            return final_response.strip()
        elif hasattr(final_response, "content"):
            return final_response.content.strip()
        else:
            return "Unexpected final response format."
    except Exception as e:
        st.error(f"Error refining query: {e}")
        return None



# Streamlit App
st.title("YouTube Transcript Processor and Query Tool")

# Input fields
video_url = st.text_input("Enter YouTube Video URL", "")
user_query = st.text_input("Enter your query/task", "")

if st.button("Process and Query"):
    if video_url and user_query:
        # Extract video ID from URL
        video_id_match = re.search(r"v=([a-zA-Z0-9_-]+)", video_url)
        if not video_id_match:
            st.error("Invalid YouTube URL.")
        else:
            video_id = video_id_match.group(1)

            # Check if chunks already exist in Pinecone
            if chunks_exist_in_pinecone(video_id):
                st.info(f"Using existing chunks for video ID {video_id}...")
            else:
                # Fetch transcript and process into chunks
                video_id, chunks = get_youtube_transcript(video_url)
                if video_id and chunks:
                    st.info(f"Fetched {len(chunks)} chunks. Storing in Pinecone...")
                    success = store_video_in_pinecone(video_id, chunks)
                    if not success:
                        st.error("Failed to store chunks in Pinecone.")
                        st.stop()

            # Process the query using existing or newly stored chunks
            st.info("Processing your query...")
            response = refine_query_with_llm(video_id, user_query)
            if response:
                st.markdown(f"### Response:\n{response}")
            else:
                st.error("Failed to process query.")
    else:
        st.warning("Please provide both a YouTube URL and a query.")
