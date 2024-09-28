import gradio as gr
import edge_tts
import asyncio
import tempfile
import numpy as np
import soxr
from pydub import AudioSegment
import torch
import sentencepiece as spm
import onnxruntime as ort
from huggingface_hub import hf_hub_download, InferenceClient
import requests
from bs4 import BeautifulSoup
import urllib
import random

# List of user agents for requests
user_agents = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:66.0) Gecko/20100101 Firefox/66.0',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36'
]

def get_useragent():
    return random.choice(user_agents)

def extract_text(html):
    """Extract visible text from HTML content."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "header", "footer", "nav"]):
        tag.extract()
    return soup.get_text(strip=True)[:8000]

def search(term, num_results=2):
    """Performs a Google search and returns the results."""
    response = requests.get(
        "https://www.google.com/search",
        headers={"User-Agent": get_useragent()},
        params={"q": term, "num": num_results}
    )
    response.raise_for_status()
    
    results = []
    for link in BeautifulSoup(response.text, "html.parser").find_all("div", class_="g"):
        url = link.find("a", href=True)
        if url:
            try:
                webpage = requests.get(url["href"], headers={"User-Agent": get_useragent()})
                webpage.raise_for_status()
                results.append({"link": url["href"], "text": extract_text(webpage.text)})
            except requests.exceptions.RequestException:
                results.append({"link": None, "text": None})
    return results

# Load models
model_name = "neongeckocom/stt_en_citrinet_512_gamma_0_25"
preprocessor = torch.jit.load(hf_hub_download(model_name, "preprocessor.ts", subfolder="onnx"))
encoder = ort.InferenceSession(hf_hub_download(model_name, "model.onnx", subfolder="onnx"))
tokenizer = spm.SentencePieceProcessor(hf_hub_download(model_name, "tokenizer.spm", subfolder="onnx"))
client = InferenceClient("HuggingFaceH4/zephyr-7b-alpha")

def resample(audio, sr):
    return soxr.resample(audio, sr, 16000)

def to_float32(audio):
    return np.divide(audio, np.iinfo(audio.dtype).max, dtype=np.float32)

def transcribe(audio_path):
    audio_file = AudioSegment.from_file(audio_path)
    sr = audio_file.frame_rate
    audio_buffer = np.array(audio_file.get_array_of_samples())
    audio_fp32 = to_float32(audio_buffer)
    audio_16k = resample(audio_fp32, sr)
    
    input_signal = torch.tensor(audio_16k).unsqueeze(0)
    length = torch.tensor(len(audio_16k)).unsqueeze(0)
    processed_signal, _ = preprocessor.forward(input_signal=input_signal, length=length)
    
    logits = encoder.run(None, {'audio_signal': processed_signal.numpy(), 'length': length.numpy()})[0][0]
    decoded_prediction = [p for p in logits.argmax(axis=1).tolist() if p != tokenizer.vocab_size()]
    return tokenizer.decode_ids(decoded_prediction)

async def respond(audio, web_search):
    user_input = transcribe(audio)
    web_results = search(user_input) if web_search else []
    web_text = ' '.join([f"Link: {res['link']}\nText: {res['text']}\n\n" for res in web_results])
    prompt = f"<s>[SYSTEM] {user_input}[WEB]{web_text}[OpenGPT 4o]"
    
    reply = "".join([response.token.text for response in client.text_generation(prompt, max_new_tokens=300, stream=True) if response.token.text != "</s>"])
    communicate = edge_tts.Communicate(reply)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        await communicate.save(tmp_file.name)
    return tmp_file.name

with gr.Blocks() as demo:
    gr.Markdown("# Emotional Support\nHello! I'm here to support you emotionally and answer any questions. How are you feeling today?")
    gr.Markdown("<p style='color:green;'>Developed by Hashir Ehtisham</p>")
    
    with gr.Row():
        web_search = gr.Checkbox(label="Web Search", value=False)
        input_audio = gr.Audio(label="User Input", sources="microphone", type="filepath")
        output_audio = gr.Audio(label="AI", autoplay=True)
        gr.Interface(fn=respond, inputs=[input_audio, web_search], outputs=[output_audio], live=True)

if __name__ == "__main__":
    demo.launch()
