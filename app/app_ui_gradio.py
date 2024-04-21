import gradio as gr
from PIL import Image
# import os
import base64
from dupchi.dupchi import DupchiAssistant
# import pandas as pd
import numpy as np
from google.cloud import storage

# def get_image_path(image_name):

#     image_path = f'./data/lung_colon_image_set_test/{image_name}'
#     return image_path

def get_image_base64(path):
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
def list_blobs(bucket_name):

    imgs_names = []

    storage_client = storage.Client.from_service_account_json('gcp/dupchi-88047a0bb300.json') # login con las credenciales de la service account
    blobs = storage_client.list_blobs(bucket_name)

    for blob in blobs:
        imgs_names.append('/'.join(blob.name.split('/')[1:]))

    return imgs_names

def download_blob(bucket_name, source_blob_name, destination_file_name):

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)

    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)


def main():
    logo_path = "./assets/dupchi_logo.png"
    idal_logo_path = "./assets/idal-logo_ph.png"

    encoded_logo = get_image_base64(logo_path)
    encoded_idal_logo = get_image_base64(idal_logo_path)

    description_html_2 =f"""
    <div style="text-align: center; margin-top: 10px;">
      <img src='data:image/png;base64,{encoded_logo}' alt='DUPCHI Logo' style='width: 100%; height: auto; margin-bottom: 10px;'>
    </div>
    """

    assistant = DupchiAssistant()

    def process_image(image_name):

        assistant.reset_process()

        image_path = f"{assistant.TMP_FOLDER_IMG}/img2process.png"

        download_blob(assistant.BUCKET_NAME, f"lung_colon_image_set_test/{image_name}", image_path)

        image = Image.open(image_path)

        labels = {'colon_aca':'Colon Adenocarcinoma', 
                'colon_n': 'Colon Benign Tissue', 
                'lung_aca': 'Lung Adenocarcinoma', 
                'lung_n': 'Lung Benign Tissue',
                'lung_scc': 'Lung Squamous Cell Carcinoma'}

        gr.Image(image, type='pil')
        
        noise_added = assistant.add_noise(image)
        noise_removed = assistant.denoise_image(noise_added)
        image_upscaled = assistant.upscale_image(noise_removed)
        class_prediction, confidence = assistant.predict_class(image_upscaled)

        noise_added_PIL = Image.fromarray((noise_added[0] * 255).astype(np.uint8))
        noise_removed_PIL = Image.fromarray((noise_removed[0] * 255).astype(np.uint8))
        # image_upscaled_PIL = Image.fromarray((image_upscaled * 255).astype(np.uint8))

        return noise_added_PIL, noise_removed_PIL, image_upscaled, labels[class_prediction], confidence

    nombres_de_imagenes = list_blobs('dupchi_test_images')

    with gr.Blocks(theme='soft', title='DUPCHI') as demo:
        gr.Markdown(description_html_2)
        gr.Markdown("# Denoise, Upscale and Predict Cancer from Histopathological Images")
        gr.Markdown(f'''
                    ---
                    <img src='data:image/png;base64,{encoded_idal_logo}' align="right" style="float" width="400">

                    ## Máster IA^3

                    ## Módulos: Aprendizaje profundo (I y II)

                    ### Javier Yago Córcoles

                    #### Data Scientist

                    ---
                    ''')
        
        with gr.Row():
            with gr.Column():
                image_selector = gr.Dropdown(choices=nombres_de_imagenes, label="Select Image")
                selected_image = gr.Image(type="pil", label="Selected Image")
                
                def update_img(change):
                    image_path = f"{assistant.TMP_FOLDER_IMG}/img2process.png"
                    download_blob(assistant.BUCKET_NAME, f"lung_colon_image_set_test/{change}", image_path)
                    image_opened = Image.open(image_path)
                    selected_image = gr.Image(value = image_opened, type="pil", label="Selected Image")

                    return selected_image
                
                image_selector.change(update_img, inputs=[image_selector], outputs=[selected_image])

                submit_button = gr.Button("Submit")                

            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        noise_added = gr.Image(type="pil", label="1- Downscaled and noise added", height=400, width=400)
                        noise_removed = gr.Image(type="pil", label="2- Noise removed", height=400, width=400)
                    with gr.Column():
                        upscaled_image = gr.Image(type="pil", label="3- Upscaled image", height=400, width=400)
                        predicted_class = gr.Textbox(label="4- Predicted Class")
                        confidence = gr.Textbox(label="Confidence")

                submit_button.click(
                    process_image,
                    inputs=[image_selector],
                    outputs=[noise_added, noise_removed, upscaled_image, predicted_class, confidence]
                )

    demo.launch(favicon_path='./assets/dupchi_favicon.png', server_name='0.0.0.0', server_port=8080)

if __name__ == "__main__":
    main()
