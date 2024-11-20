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


def download_file(url, destination):
    directory = os.path.dirname(destination)
    if directory:
        os.makedirs(directory, exist_ok=True)
    if not os.path.exists(destination):
        gdown.download(url, destination, quiet=False)


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
        download_file(checkpt_url, checkpt_path)
        download_file(char_map_url, char_map_path)
        download_file(lexicon_url, lexicon_file)

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

    def _stitch_images(self, images, padding=-4):
        """
        Stitches a list of word images together horizontally to create a full line image.
        
        :param images: List of word images to stitch.
        :param padding: Padding between images. If negative, images will overlap.
        """
        if len(images) == 0:
            return np.full((32, 32), fill_value=255, dtype=np.uint8)

        total_width = sum(img.shape[1] for img in images) + padding * (len(images) - 1)
        max_height = max(img.shape[0] for img in images)
        stitched_image = np.full((max_height, total_width), fill_value=255, dtype=np.uint8)

        current_x = 0
        for img in images:
            stitched_image[:img.shape[0], current_x:current_x + img.shape[1]] = img
            current_x += img.shape[1] + padding

        return stitched_image

    def generate_image(self, text, seed=None):
        """
        Generates an image for a given word or a sequence of words using the ImgGenerator.
        """
        if not isinstance(text, str):
            raise ValueError("Input must be a string representing a word or sentence.")

        if text == "":
            # Return an empty white image for empty strings
            return np.full((32, 32), fill_value=255, dtype=np.uint8)

        # Split text into words if it contains spaces
        words = text.split()

        # Set random seed for reproducibility
        seed = random.randint(0, 10000) if seed is None else seed
        torch.manual_seed(seed)
        z = self.z_dist.sample([128]).unsqueeze(0)

        # Generate images for each word
        generated_imgs, _, _ = self.img_generator.generate(word_list=words, z=z)
        word_images = [self._rescale_image(img) for img in generated_imgs]

        # Stitch word images together if there are multiple words
        if len(word_images) > 1:
            return self._stitch_images(word_images)
        else:
            return word_images[0]

    def generate_empty_row(self, width=1225, height=84):
        """
        Generates an empty row image without vertical lines for columns.
        """
        return Image.new('L', (width, height), color=255)

    def add_image_to_row(self, row_image, word_image, x_offset, y_offset):
        """
        Adds a word image to a specific cell in the row image.
        """
        row_pil = Image.fromarray(row_image)
        word_pil = Image.fromarray(word_image)

        # Calculate position for the word image within the row
        x = int(max(0, x_offset))
        y = int(max(0, (row_image.shape[0] - word_image.shape[0]) // 2 + y_offset))

        # Paste the word image onto the row image
        row_pil.paste(word_pil, (x, y), word_pil if word_pil.mode == 'RGBA' else None)
        return np.array(row_pil)

    def draw_vertical_lines(self, row_image, column_structure):
        """
        Draw vertical lines for column separators after adding the word images.
        """
        row_pil = Image.fromarray(row_image)
        draw = ImageDraw.Draw(row_pil)

        # Draw vertical lines for column separators
        if column_structure is not None:
            x_offset = 0
            for col_width in column_structure:
                x_offset += col_width
                draw.line([(x_offset, 0), (x_offset, row_pil.height)], fill=0)

        return np.array(row_pil)

    def generate_row(self, row_height, row_width, columns, seed=None, background_color=255):
        """
        Generates a row of images for a list of column dictionaries, concatenated horizontally with an empty row background.
        """
        row_image = np.array(self.generate_empty_row(width=row_width, height=row_height))

        seed = random.randint(0, 10000) if seed is None else seed
        random.seed(seed)

        x_offset = 0
        for column in columns:
            text = column.get('content', '')
            x_translation = column.get('x_translation', 0)
            y_translation = column.get('y_translation', 0)
            resize_factor = column.get('resize_factor', 1.0)

            word_image = self.generate_image(text, seed=seed)
            
            if word_image is not None:
                if resize_factor != 1.0:
                    word_image = self._resize_image(word_image, factor=resize_factor)

                row_image = self.add_image_to_row(row_image, word_image, x_offset + x_translation, y_translation)

            x_offset += column['width']

        row_image = self.draw_vertical_lines(row_image, [col['width'] for col in columns])
        return row_image

    def generate_rows(self, column_structure, table):
        """
        Generates a row image for each row in the dataframe and saves it to the path specified in the 'image_path' column.
        """
        for index, row in table.iterrows():
            columns = []
            for col in column_structure:
                content = row[col['name']] if col['name'] in row else ''
                x_translation = random.uniform(-col.get('x_translation_max', -2), col.get('x_translation_max', 2))
                y_translation = random.uniform(col.get('y_translation_min', 2), -col.get('y_translation_min', -2))
                resize_factor = random.uniform(col.get('resize_factor_min', 1.0), col.get('resize_factor_max', 1.7))
                columns.append({
                    'name': col['name'],
                    'width': col['width'],
                    'content': content,
                    'x_translation': int(x_translation),
                    'y_translation': int(y_translation),
                    'resize_factor': resize_factor
                })

            row_image = self.generate_row(row_height=84, row_width=sum([col['width'] for col in column_structure]), columns=columns)
            image_path = row['image_path']
            Image.fromarray(row_image).save(image_path)


if __name__ == "__main__":
    # Create an instance of DataGenerator
    data_gen = DataGenerator(dataset='IAM')

    # Define column properties as dictionaries
    column_structure = [
        {'name': 'NOMS DE FAMILLE', 'width': 220},
        {'name': 'PRENOMS', 'width': 186},
        {'name': 'ANNEE de NAISSANCE', 'width': 80},
        {'name': 'LIEU de NAISSANCE', 'width': 158},
        {'name': 'NATIONALITE', 'width': 105},
        {'name': 'ETAT MATRIMONIAL', 'width': 38},
        {'name': 'SITUATION PAR RAPPORT au chef de ménage', 'width': 110},
        {'name': 'DEGRE D INSTRUCTION', 'width': 38},
        {'name': 'PROFESSION', 'width': 156},
        {'name': 'REMARQUES', 'width': 134}
    ]

    # Create a sample DataFrame
    data = {
        'NOMS DE FAMILLE': ['Levy marie'],
        'PRENOMS': ['Charles'],
        'ANNEE de NAISSANCE': ['76'],
        'LIEU de NAISSANCE': ['Paris'],
        'NATIONALITE': ['Francaise'],
        'ETAT MATRIMONIAL': ['m'],
        'SITUATION PAR RAPPORT au chef de ménage': ['Chef'],
        'DEGRE D INSTRUCTION': [''],
        'PROFESSION': ['hotelier'],
        'REMARQUES': [''],
        'image_path': ['generated_row_image_1.png']
    }
    table = pd.DataFrame(data)

    # Generate rows and save the images
    data_gen.generate_rows(column_structure=column_structure, table=table)
