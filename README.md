
# YouTube Transcript Query Tool

A web application built with Streamlit to process YouTube transcripts, store them in Pinecone, and query or summarize information using OpenAI's GPT-3.5.

## Features

- Fetch YouTube video transcripts automatically.
- Store transcript chunks in Pinecone for efficient retrieval.
- Query the transcript using OpenAI's GPT-3.5 for summarization, Q&A, or bullet points.
- Automatic chunking to handle token limits.
- Simple and interactive Streamlit-based user interface.

## Technologies Used

- **Streamlit**: User interface framework.
- **OpenAI GPT-3.5**: Natural language processing.
- **Pinecone**: Scalable vector database.
- **YouTube Transcript API**: Fetches transcripts.

## Prerequisites

- Python 3.9+ installed on your system.
- API keys:
  - **OpenAI API Key**: [Get it here](https://platform.openai.com/).
  - **Pinecone API Key**: [Get it here](https://www.pinecone.io/).

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/youtube-transcript-query-tool.git
   cd youtube-transcript-query-tool
   ```

2. Install the required Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set environment variables for OpenAI and Pinecone:
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   export PINECONE_API_KEY="your-pinecone-api-key"
   ```

## Running the App

Start the Streamlit app using the command:
```bash
streamlit run app.py
```

Open the URL displayed in your terminal (default: `http://localhost:8501`) in your browser.

## Usage

1. Enter the YouTube video URL in the input field.
2. Enter your task/query (e.g., "Summarize the video" or "Give me 5 important points").
3. Click **"Process and Query"**.
4. View the response generated by the app.

## Deployment

### Streamlit Community Cloud
1. Push your project to a GitHub repository.
2. Go to [Streamlit Cloud](https://share.streamlit.io/).
3. Select your GitHub repository and the `app.py` file.
4. Deploy the app and share the public URL.

### Heroku
1. Install the Heroku CLI.
2. Add a `Procfile` with the following content:
   ```plaintext
   web: streamlit run app.py --server.port=$PORT --server.headless=true
   ```
3. Push your code to Heroku and deploy.

For other deployment options like AWS or Docker, see detailed documentation online.

## Limitations

- Processing large transcripts may take longer.
- Limited to OpenAI's GPT-3.5 model's maximum token context (4097 tokens).
- Requires valid YouTube video transcripts.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
![Diagram](diagram.png)

## Contact

- **Author**: Sai Teja
- **Email**: saiteja.srivilli@gmail.com
- **GitHub**: [saitejasrivilli](https://github.com/saitejasrivilli)
