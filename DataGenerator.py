import pickle as pkl
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
from importlib import import_module
from config import Config
from utils.data_utils import WordMap
from utils.training_utils import ModelCheckpoint
from generate_images import ImgGenerator
import random
from PIL import Image, ImageEnhance, ImageDraw
import os
import requests
import gdown

class DataGenerator:
    def __init__(self, dataset='IAM', config=Config, device='cpu'):
        """
        Initializes the DataGenerator class by loading the character map and model.
        """
        # Set dataset-specific paths and download if necessary
        if dataset == 'IAM':
            checkpt_path = 'IAM_best_checkpoint.tar'
            char_map_path = 'IAM_char_map.pkl'
            lexicon_file = 'data/Lexicon/words.txt'
            checkpt_url = 'https://drive.google.com/uc?id=11w1p8RVLml9cidMrkQpdo648pNPdQFxZ'
            char_map_url = 'https://drive.google.com/uc?id=10bXCFp7a7MyUFKR55rUeqWNrEe2300TP'
            lexicon_url = 'https://raw.githubusercontent.com/dwyl/english-words/master/words.txt'
        elif dataset == 'RIMES':
            checkpt_path = 'RIMES_best_checkpoint.tar'
            char_map_path = 'RIMES_char_map.pkl'
            lexicon_file = 'data/Lexicon/Lexique383.tsv'
            checkpt_url = 'https://drive.google.com/uc?id=16oasVsBExwHhCmYDSR1uhV10NiWYZ-OY'
            char_map_url = 'https://drive.google.com/uc?id=1vjj7DfT_T3c4q-18LNpo7YMcRCUnh7aF'
            lexicon_url = 'https://raw.githubusercontent.com/AdrienVannson/Decorrecteur/refs/heads/master/Lexique383'
        else:
            raise ValueError("Unsupported dataset. Choose 'IAM' or 'RIMES'.")

        # Download files if they do not exist
        self.download_file(checkpt_url, checkpt_path)
        self.download_file(char_map_url, char_map_path)
        self.download_file(lexicon_url, lexicon_file)

        # Load character map
        with open(char_map_path, 'rb') as f:
            char_map = pkl.load(f)
        config.device = device
        config.dataset = dataset
        config.lexicon_file = lexicon_file
        config.num_chars = 74 if dataset == 'IAM' else 93

        # Initialize ImgGenerator
        self.img_generator = ImgGenerator(checkpt_path=checkpt_path, config=config, char_map=char_map)
        self.z_dist = torch.distributions.Normal(loc=0, scale=1.)

    @staticmethod
    def download_file(url, destination):
        directory = os.path.dirname(destination)
        if directory:
            os.makedirs(directory, exist_ok=True)
        if not os.path.exists(destination):
            gdown.download(url, destination, quiet=False)

    def _rescale_image(self, img):
        """
        Rescales an image from range [-1, 1] to [0, 255].
        """
        return ((img + 1) * 127.5).clip(0, 255).astype(np.uint8)

    def _resize_image(self, img, factor=1.2):
        """
        Resizes the given image by a specific factor.
        """
        new_width = int(img.shape[1] * factor)
        new_height = int(img.shape[0] * factor)
        return np.array(Image.fromarray(img).resize((new_width, new_height), Image.LANCZOS))

    def generate_image(self, text, seed=None):
        """
        Generates an image for a given text, splitting into words if there are spaces.
        """
        if not isinstance(text, str):
            raise ValueError("Input must be a string.")

        if text == "":
            # Return an empty white image for empty strings
            return np.full((32, 32), fill_value=255, dtype=np.uint8)

        # Split text by spaces to generate each word separately
        words = text.split()

        # Set random seed for reproducibility
        seed = random.randint(0, 10000) if seed is None else seed
        torch.manual_seed(seed)
        z = self.z_dist.sample([128]).unsqueeze(0)

        # Generate images for each word
        generated_imgs, _, _ = self.img_generator.generate(word_list=words, z=z)
        
        # Concatenate all generated word images horizontally
        word_images = [self._rescale_image(img) for img in generated_imgs]
        concatenated_image = self.concatenate_word_images(word_images)

        return concatenated_image

    def concatenate_word_images(self, word_images, space_width=20):
        """
        Concatenates word images with spaces between them.

        :param word_images: List of word images to concatenate.
        :param space_width: Width of the space between words.
        :return: Concatenated image.
        """
        # Determine total width and height
        total_width = sum(img.shape[1] for img in word_images) + space_width * (len(word_images) - 1)
        max_height = max(img.shape[0] for img in word_images)

        # Create an empty white image to serve as the background
        concatenated_image = np.full((max_height, total_width), fill_value=255, dtype=np.uint8)

        # Paste each word image onto the concatenated image
        x_offset = 0
        for word_img in word_images:
            concatenated_image[:word_img.shape[0], x_offset:x_offset + word_img.shape[1]] = word_img
            x_offset += word_img.shape[1] + space_width

        return concatenated_image

    def generate_rows(self, column_structure, table):
        """
        Generates a row image for each row in the dataframe and saves it to the path specified in the 'image_path' column.
        """
        for index, row in table.iterrows():
            columns = []
            for col in column_structure:
                content = row[col['name']] if col['name'] in row else ''
                x_translation = random.uniform(-col.get('x_translation_max', -2), col.get('x_translation_max', 2))
                x_translation = int(x_translation)
                y_translation = random.uniform(col.get('y_translation_min', 2), -col.get('y_translation_min', -2))
                y_translation = int(y_translation)
                resize_factor = random.uniform(col.get('resize_factor_min', 1.0), col.get('resize_factor_max', 1.7))
                columns.append({
                    'name': col['name'],
                    'width': col['width'],
                    'content': content,
                    'x_translation': x_translation,
                    'y_translation': y_translation,
                    'resize_factor': resize_factor
                })

            # Generate the row image
            row_image = self.generate_row(row_height=84, row_width=sum([col['width'] for col in column_structure]), columns=columns)

            # Save the final processed row image
            image_path = row['image_path']
            Image.fromarray(row_image).save(image_path)

