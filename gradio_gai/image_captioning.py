import os
import io
from transformers import pipeline
import gradio as gr 

get_completion = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

def captioner(image):
    result = get_completion(image)
    return result[0]['generated_text']

demo = gr.Interface(fn=captioner,
                    inputs=[gr.Image(label="Upload image", type="pil")],
                    outputs=[gr.Textbox(label="Caption")],
                    title="Image Captioning with BLIP",
                    description="Caption any image using the BLIP model",
                    allow_flagging="never",
                    examples=[])

demo.launch(share=True, server_port=int('8088'))
