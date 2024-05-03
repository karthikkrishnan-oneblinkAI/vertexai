import vertexai
from vertexai.generative_models import GenerativeModel, ChatSession
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput
from typing import List

import requests

def get_project_id():
    try:
        url = "http://metadata.google.internal/computeMetadata/v1/project/project-id"
        headers = {"Metadata-Flavor": "Google"}
        response = requests.get(url, headers=headers)
        return response.text
    except Exception as e:
        print("Error getting project ID. May be not on GCP VM?")
        project_id = input("Enter the Google Cloud project ID: ")
        return project_id


class VertexAISmokeTester:
    def __init__(self, project_id: str, location: str = 'us-central1'):
        self.project_id = project_id
        self.location = location
        vertexai.init(project=self.project_id, location=self.location)
        self.model = GenerativeModel(model_name="gemini-1.0-pro-002")
        self.chat = self.model.start_chat()

    def get_chat_response(self, prompt: str) -> str:
        responses = self.chat.send_message(prompt, stream=True)
        text_response = []
        for chunk in responses:
            text_response.append(chunk.text)
        return "".join(text_response)

    def create_embeddings(self, texts: List[str], task: str, model_name: str) -> List[List[float]]:
        model = TextEmbeddingModel.from_pretrained(model_name)
        inputs = [TextEmbeddingInput(text, task) for text in texts]
        embeddings = model.get_embeddings(inputs)
        return [embedding.values for embedding in embeddings]

# Example usage:
if __name__ == "__main__":
    #project_id = "precious-sky-23422"  # Update with your project ID
    # take input
    project_id = get_project_id()
    tester = VertexAISmokeTester(project_id=project_id)

    # Test Chat Response
    prompts = ["Hello.", "What are all the colors in a rainbow?", "Why does it appear when it rains?"]
    for prompt in prompts:
        print(f"Prompt: '{prompt}' Response: '{tester.get_chat_response(prompt)}'")

    # Test Creating Embeddings
    texts = ["Hello, how are you?", "Weather is great today!"]
    embeddings = tester.create_embeddings(texts, "RETRIEVAL_DOCUMENT", "textembedding-gecko@003")
    print(f"Embeddings: {embeddings}")
