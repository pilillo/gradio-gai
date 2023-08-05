import os
import io
from transformers import pipeline
import gradio as gr 

get_completion = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def summarizer(text):
    result = get_completion(text)
    return result[0]['summary_text']

demo = gr.Interface(fn=summarizer,
                    inputs=[gr.Textbox(label="Text", lines=6)],
                    outputs=[gr.Textbox(label="Summary", lines=3)],
                    title="Text Summarization with distillbart-cnn",
                    description="Summarize any text using the `sshleifer/distilbart-cnn-12-6` ",
                    allow_flagging="never",
                    )

demo.launch(share=True, server_port=int('8088'))
