import os
import io
import gradio as gr
from diffusers import DiffusionPipeline
#pipeline = DiffusionPipeline.from_pretrained("CompVis/ldm-text2im-large-256")
pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

def generator(text, negative_prompt, steps, guidance, width, height):
    result = pipeline(text, negative_prompt=negative_prompt, guidance_scale=guidance, num_inference_steps=steps, width=width, height=height).images[0]
    return result

demo = gr.Interface(fn=generator,
                    inputs=[
                        gr.Textbox(label="Image caption"),
                        gr.Textbox(label="Negative prompt"),
                        gr.Slider(label="Inference steps", minimum=1, maximum=100, step=1, value=25, info="Controls how many steps the denoiser does denoise the image"),
                        gr.Slider(label="Guidance scale", minimum=1, maximum=20, step=1, value=7, info="Controls how much the text prompt influences the result"),
                        gr.Slider(label="Width", minimum=64, maximum=512, step=64, value=256),
                        gr.Slider(label="Height", minimum=64, maximum=512, step=64, value=256),
                    ],
                    outputs=[gr.Image(label="Result", type="pil")],
                    title="Image Generation with Stable Diffusion",
                    description="Generate an image from a text caption using the stable diffusion model",
                    allow_flagging="never",
                    examples=[])

demo.launch(share=True, server_port=int('8088'))
