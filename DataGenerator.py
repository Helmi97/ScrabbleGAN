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

def download_file(url, destination):
    if not os.path.exists(destination):
        response = requests.get(url)
        response.raise_for_status()  # Raise an error on bad status
        with open(destination, 'wb') as f:
            f.write(response.content)

class DataGenerator:
    def __init__(self, dataset='IAM', config=Config, device='cpu'):
        """
        Initializes the DataGenerator class by loading the character map and model.

        :param dataset: Name of the dataset ('IAM' or 'RIMES').
        :param config: Config object containing model parameters.
        :param device: Device on which the model should run ('cpu' or 'cuda').
        """
        # Set dataset-specific paths and download if necessary
        if dataset == 'IAM':
            checkpt_path = 'IAM_best_checkpoint.tar'
            char_map_path = 'IAM_char_map.pkl'
            lexicon_file = 'data/Lexicon/words.txt'
            checkpt_url = 'https://drive.google.com/uc?id=11w1p8RVLml9cidMrkQpdo648pNPdQFxZ'
            char_map_url = 'https://drive.google.com/uc?id=10bXCFp7a7MyUFKR55rUeqWNrEe2300TP'
            lexicon_url = 'https://github.com/dwyl/english-words/blob/master/words.txt'
        elif dataset == 'RIMES':
            checkpt_path = 'RIMES_best_checkpoint.tar'
            char_map_path = 'RIMES_char_map.pkl'
            lexicon_file = 'data/Lexicon/Lexique383.tsv'
            checkpt_url = 'https://drive.google.com/uc?id=16oasVsBExwHhCmYDSR1uhV10NiWYZ-OY'
            char_map_url = 'https://drive.google.com/uc?id=1vjj7DfT_T3c4q-18LNpo7YMcRCUnh7aF'
            lexicon_url = 'https://github.com/AdrienVannson/Decorrecteur/blob/master/Lexique383'
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

        :param img_array: Numpy array representing the image.
        :return: Rescaled image in numpy array format.
        """
        return ((img + 1) * 127.5).clip(0, 255).astype(np.uint8)
    
    def _resize_image(self, img, factor=1.2):
        """
        Resizes the given image by a specific factor.

        :param img: Numpy array representing the image.
        :param factor: The factor by which to resize the image.
        :return: Resized image.
        """
        new_width = int(img.shape[1] * factor)
        new_height = int(img.shape[0] * factor)
        return np.array(Image.fromarray(img).resize((new_width, new_height), Image.LANCZOS))


    def generate_image(self, text, seed=None):
        """
        Generates an image for a given word using the ImgGenerator.

        :param text: A single word as a string.
        :param seed: Seed for generating style variations. If None, a random seed is used.
        :return: Generated image in numpy array format.
        """
        if not isinstance(text, str):
            raise ValueError("Input must be a string representing a single word.")

        if text == "":
            # Return an empty white image for empty strings
            return np.full((32, 32), fill_value=255, dtype=np.uint8)

        # Set random seed for reproducibility
        seed = random.randint(0, 10000) if seed is None else seed
        torch.manual_seed(seed)
        z = self.z_dist.sample([128]).unsqueeze(0)

        # Generate image
        generated_imgs, _, _ = self.img_generator.generate(word_list=[text], z=z)
        return self._rescale_image(generated_imgs[0]) if generated_imgs.size > 0 else None

    def generate_empty_row(self, width=1225, height=84, column_structure=None):
        """
        Generates an empty row image without vertical lines for columns.

        :param width: Width of the entire row.
        :param height: Height of the row.
        :param column_structure: List of column widths.
        :return: Empty row image.
        """
        # Create an empty white image
        return Image.new('L', (width, height), color=255)

    def add_image_to_row(self, row_image, word_image, x_offset, y_offset):
        """
        Adds a word image to a specific cell in the row image.

        :param row_image: The empty row image without column separators.
        :param word_image: The generated word image to be added.
        :param x_offset: Horizontal offset for positioning the word.
        :param y_offset: Vertical offset for positioning the word.
        :return: The updated row image with the word added.
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

        :param row_image: The row image after adding word images.
        :param column_structure: List of column widths.
        :return: Row image with vertical lines added.
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
        Adds images to each cell with specified offsets for realism.

        :param row_height: Height of the entire row image.
        :param row_width: Width of the entire row image.
        :param columns: List of dictionaries defining each column's properties ('name', 'width', 'content', etc.).
        :param seed: Seed for generating style variations.
        :param background_color: Background color for padding (default is 255 for white).
        :return: Final row image with words added and vertical lines drawn.
        """
        # Generate an empty row without vertical lines
        row_image = np.array(self.generate_empty_row(width=row_width, height=row_height))

        # Set random seed for reproducibility
        seed = random.randint(0, 10000) if seed is None else seed
        random.seed(seed)

        # Generate and add images for each column's content in the corresponding cell
        x_offset = 0
        for column in columns:
            text = column.get('content', '')
            x_translation = column.get('x_translation', 0)
            y_translation = column.get('y_translation', 0)
            resize_factor = column.get('resize_factor', 1.0)

            # Generate word image
            word_image = self.generate_image(text, seed=seed)
            
            if word_image is not None:
                # Resize image if resize factor is provided
                if resize_factor != 1.0:
                    word_image = self._resize_image(word_image, factor=resize_factor)

                # Add word image to the appropriate cell in the row
                row_image = self.add_image_to_row(row_image, word_image, x_offset + x_translation, y_translation)

            # Update x_offset based on column width
            x_offset += column['width']

        # Draw vertical lines after adding word images
        row_image = self.draw_vertical_lines(row_image, [col['width'] for col in columns])

        return row_image

    def generate_rows(self, column_structure, table):
        """
        Generates a row image for each row in the dataframe and saves it to the path specified in the 'image_path' column.

        :param column_structure: List of dictionaries representing the column structure.
        :param table: A pandas DataFrame containing one row for each image, with a column 'image_path' specifying where to save the image.
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
        'NOMS DE FAMILLE': ['Levy'],
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
