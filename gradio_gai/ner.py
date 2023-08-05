import os
import io
from transformers import pipeline
import gradio as gr 

get_completion = pipeline("ner", model="dslim/bert-base-NER")

def ner(text):
    result = get_completion(text)
    return {'text':text, 'entities':result}

demo = gr.Interface(fn=ner,
                    inputs=[gr.Textbox(label="Text", lines=6)],
                    outputs=[gr.HighlightedText(label="Text with entities")],
                    title="NER with distillbart-cnn",
                    description="NER on any text using the `dslim/bert-base-NER` ",
                    allow_flagging="never",
                    examples=["The pen is on the table"]
                    )

demo.launch(share=True, server_port=int('8088'))
