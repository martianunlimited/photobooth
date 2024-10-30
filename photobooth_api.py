import gradio as gr
import requests
import base64
from PIL import Image, ImageEnhance, ImageFilter
from io import BytesIO
import numpy as np
import time
import threading

import os

# Function to get base64 representation of an image
def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Get the base64 string of the image stored in the same folder
base64_image_string = get_base64_image(os.path.join(os.path.dirname(__file__), 'image.gif'))

# API URL
API_URL = "http://127.0.0.1:7860/sdapi/v1/control"

# Function to handle image resizing and processing
def process_image(input_image, description):
    if input_image is None:
        print("No image captured from the webcam!")
        return input_image

    # Resize the input image so the shortest side is at most 512
    max_size = 512
    width, height = input_image.size
    if width < height:
        new_width = max_size
        new_height = int((max_size / width) * height)
    else:
        new_height = max_size
        new_width = int((max_size / height) * width)

    input_image = input_image.resize((new_width, new_height))
    mean = 0
    sigma = 15
    g_noise = np.random.normal(mean, sigma, (new_height, new_width, 3))
    n_image = np.clip(input_image + g_noise, 0, 255).astype(np.uint8)
    n_image = Image.fromarray(n_image)

    # Convert the resized image to base64
    buffered = BytesIO()
    input_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

    buffered = BytesIO()
    n_image.save(buffered, format="PNG")
    n_img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # Set strength based on description
    strength_mapping = {
        "zombie": 0.6,
        "skeleton": 0.6,
        "knight": 0.6,
        "orc": 0.7,
        "goblin": 0.6,
        "angel": 0.75,
        "demon": 0.7,
        "vampire": 0.7,
        "elf": 0.6,
        "space marine": 0.7
    }
    strength = strength_mapping.get(description, 0.5)

    # Prepare the payload with the image and additional data
    prompt_mapping = {
        "zombie": "Imagine yourself as a scary undead creature, a zombie, with tattered clothes.",
        "skeleton": "Imagine yourself as a spooky skeleton, with visible bones and hollow eyes.",
        "knight": "Imagine yourself as a brave medieval knight, wearing shiny helmet and armor.",
        "orc": "Imagine yourself as a fierce orc, with green skin, pointed teeth and a muscular build.",
        "goblin": "Imagine yourself as a mischievous goblin, with pointy ears.",
        "angel": "Imagine yourself as an angel, with radiant wings and a halo, exuding peace and light.",
        "demon": "Imagine yourself as a powerful red devil, with dark wings, horns, and a fiery aura.",
        "vampire": "Imagine yourself as a vampire, with fangs, and a pale face.",
        "elf": "Imagine yourself as an elegant elf, with pointed ears.",
        "space marine": "Imagine yourself as a space marine in full helmet and armor, ready for battle."
    }
    prompt = prompt_mapping.get(description, description)

    payload = {
        "units": ["IP2P"],
        "inputs": [n_img_str],
        "inits": [img_str],
        "prompt": prompt,
        "unit_type": "controlnet",
        "negative_prompt": "nsfw, nudity, female_nudity, topless",
        "steps": 15,
        "cfg_scale": 6,
        "clip_skip": 1,
        "image_cfg_scale": 6,
        "width_before": new_width,
        "height_before": new_height,
        "denoising_strength": strength,
        "seed": -1
    }

    # Send the image and additional data to the API
    response = requests.post(API_URL, json=payload)
    response_data = response.json()

    # Decode the base64 output image from the API
    output_image_data = base64.b64decode(response_data['images'][0])
    output_image = Image.open(BytesIO(output_image_data))

    return output_image

# Gradio interface
def create_interface():
    with gr.Blocks(theme="gstaff/xkcd") as interface:
        header = gr.HTML("""
               <div style="padding: 20px; background-color: #000000; border-radius: 10px;">
                  <h1 style="font-size: xxx-large">AI Photobooth (Halloween Edition)</h1>
               </div>
               """)
        with gr.Row() as mainRow:
            with gr.Column(scale=3):
                with gr.Row():
                    # Create an image input and output widget with webcam source
                    input_image = gr.Image(height=480, width=640, source='webcam', streaming=True, type="pil", label="Input Image")
                    output_image = gr.Image(type="pil", label="Output Image")
                
                    # Add instructions panel on the right
    
            # Additional fields to be sent in the request
                description = gr.Radio(choices=["zombie", "skeleton", "orc", "goblin","demon", "angel", "vampire","knight","elf"], label="What you would like to be imagined as?", value="zombie", elem_id="transformation_radio")
    
            # Create a button to trigger the process
                button = gr.Button("Process Image", elem_id="process_button")
    
            # Link the button with the image processing function
                button.click(fn=process_image, inputs=[input_image, description], outputs=output_image)
            with gr.Column(scale=1):
                instruct_html=f'''
                    <div style="padding: 20px; background-color: #000000; border-radius: 10px;">
                        <h2 style="font-size: xx-large; text-decoration: underline;">Instructions:</h2>
                        <ol style="font-size: x-large;">
                            <li>Select what you would like to be imagined as from the options.</li>
                            <li>Stand in front of the camera.</li>
                            <li>Step on the foot pedal and wait approximately 10 seconds.</li>
                        </ol>
                        <h2>How Stable Diffusion Works:</h2>
                        <p>Stable Diffusion is an AI model that generates images by gradually refining random noise to match the given description. It uses a process called diffusion to iteratively remove noise and create coherent images based on the provided prompt.</p>
                        <p id='base64-image-placeholder'><img src='data:image/gif;base64,{base64_image_string}'></p>
                    </div>'''
                instructions = gr.HTML(instruct_html)

        # Add custom JavaScript for keyboard event listener
        interface.load(_js="""
        function() {
            document.addEventListener('keydown', function(event) {
                if (event.ctrlKey && event.altKey && event.key === 'p') {
                    let processButton = document.getElementById('process_button');
                    if (processButton) {
                        processButton.click();
                    }
                }
            });
            
            // Add custom styling to radio buttons to make them look like toggle buttons
            const style = document.createElement('style');
            style.innerHTML = `
                #transformation_radio input[type="radio"] {
                    display:none;
                }
                #transformation_radio label {
                    font-size: 16px;
                    border: 2px solid transparent;
                    padding: 10px;
                    cursor: pointer;
                    transition: all 0.3s ease;
                    display: inline-block;
                    margin: 5px;
                    color: #ffffff;
                    width: 100px;
                }
                #transformation_radio label:hover {
                    border-color: #3b0000;
                }
                #transformation_radio label:has(input[type="radio"]:checked) {
                    border-color: #3b0000;
                    background-color: orange;
                    color: #3b0000; /* Very dark maroon */
                    font-weight: bold;
                }
            `;
            document.head.appendChild(style);
        }
        """)
    return interface, button, input_image, output_image, description

# Launch the interface
interface, button_widget, input_image_widget, output_image_widget, description_widget = create_interface()

interface.launch()
