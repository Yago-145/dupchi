from PIL import Image
import os
# import base64
import numpy as np
import keras
from dupchi.models.ESRGAN.predict import upscale_images
# import time

class DupchiAssistant:

    TMP_FOLDER_HR = 'temp_hr'
    TMP_FOLDER_LR = 'temp_lr'
    TMP_FOLDER_IMG = 'temp_img'
    BUCKET_NAME = 'dupchi_test_images'

    def __init__(self):

        os.makedirs(self.TMP_FOLDER_HR, exist_ok=True)
        os.makedirs(self.TMP_FOLDER_LR, exist_ok=True)
        os.makedirs(self.TMP_FOLDER_IMG, exist_ok=True)

        self.autoencoder = keras.models.load_model('./model_checkpoints/autoencoder.keras')
        self.densenet = keras.models.load_model('./model_checkpoints/cnn.keras')

        pass

    def reset_process(self):
        """
        Clean the temporary folders
        """

        for file in os.listdir(self.TMP_FOLDER_HR):
            os.remove(os.path.join(self.TMP_FOLDER_HR, file))

        for file in os.listdir(self.TMP_FOLDER_LR):  
            os.remove(os.path.join(self.TMP_FOLDER_LR, file))

        # for file in os.listdir(self.TMP_FOLDER_IMG):  
        #     os.remove(os.path.join(self.TMP_FOLDER_IMG, file))

        pass

    def add_noise(self, image):
        """
        Process the image by adding noise
        """

        image_np = np.array(image.resize((224, 224), resample=Image.BICUBIC).convert("RGB"))

        noise = np.random.normal(0, 0.09, image_np.shape)
        noisy_image_np = image_np + noise * 255
        noisy_image_np = np.clip(noisy_image_np, 0, 255).astype(np.uint8)

        noisy_image = noisy_image_np.astype('float16') / 255.
        noisy_image = np.reshape(noisy_image, (1, 224, 224, 3))

        return noisy_image

    def denoise_image(self, image):
        """
        Denoise the image with the trained autoencoder
        """

        denoised_image = self.autoencoder.predict(image)

        return denoised_image

    def upscale_image(self, image):
        """
        Upscale the image with the pre-trained ESRGAN
        """

        image = Image.fromarray((image[0] * 255).astype(np.uint8))
        image.save(f'{self.TMP_FOLDER_LR}/temp.png')

        upscale_images(self.TMP_FOLDER_LR, self.TMP_FOLDER_HR)

        upscaled_image = Image.open(f'{self.TMP_FOLDER_HR}/temp.png')

        return upscaled_image

    def predict_class(self, image):
        """
        Classify the image with the fine-tuned DenseNet
        """

        image_array = np.array(image)
        image_array = image_array.astype('float16') / 255.
        image_array = np.reshape(image_array, (1, 448, 448, 3))
        
        prediction = self.densenet.predict(image_array)

        # decode the prediction
        class_labels = ['colon_aca', 'colon_n', 'lung_aca', 'lung_n', 'lung_scc']
        top_3_indices = np.argsort(prediction[0])[-3:][::-1]
        top_3_values = prediction[0][top_3_indices]
        top_3_labels = [class_labels[i] for i in top_3_indices]

        for label, value in zip(top_3_labels, top_3_values):
            print(f"{label}: {value:.4f}")

        return top_3_labels[0], top_3_values[0]
    