!pip install gradio
!pip install opena
import gradio as gr
import openai
!pip install --upgrade gradio

# Set your OpenAI API key
openai.api_key = "#"

def transcribe_audio(audio_path):
    with open(audio_path, "rb") as audio_file:
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
    return transcript["text"]

def summarize_text(text):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that summarizes text."},
            {"role": "user", "content": f"Summarize the following text:\n\n{text}"}
        ],
        max_tokens=100
    )
    return response.choices[0].message["content"].strip()

def analyze_fluency(text):
    words = text.split()
    word_count = len(words)
    # Assuming an average speaking rate of 150 words per minute
    estimated_duration = word_count / 2.5  # 2.5 words per second
    fluency_score = word_count / estimated_duration if estimated_duration > 0 else 0
    return f"Estimated Fluency Score: {fluency_score:.2f} words per second"

def process_audio(audio):
    if audio is None:
        return "No audio file provided.", "", ""
    
    transcript = transcribe_audio(audio)
    summary = summarize_text(transcript)
    fluency_analysis = analyze_fluency(transcript)
    return transcript, summary, fluency_analysis

with gr.Blocks() as demo:
    gr.Markdown("# Audio Analysis App")
    
    with gr.Row():
        audio_input = gr.Audio(type="filepath", label="Upload Audio")
    
    with gr.Row():
        transcript_output = gr.Textbox(label="Transcript")
        summary_output = gr.Textbox(label="Summary")
        fluency_output = gr.Textbox(label="Fluency Analysis")
    
    submit_button = gr.Button("Analyze")
    submit_button.click(process_audio, inputs=audio_input, outputs=[transcript_output, summary_output, fluency_output])

if __name__ == "__main__":
    demo.launch()
