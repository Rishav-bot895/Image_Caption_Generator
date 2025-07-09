import os
import pickle
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model, Model
from PIL import Image
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageCaptionGenerator:
    def __init__(self, model_path, tokenizer_path, max_length=35):
        """
        Initialize the Image Caption Generator
        
        Args:
            model_path (str): Path to the trained caption model
            tokenizer_path (str): Path to the tokenizer pickle file
            max_length (int): Maximum length of generated captions
        """
        self.max_length = max_length
        self.caption_model = None
        self.tokenizer = None
        self.vgg_model = None
        
        try:
            # Load the trained caption model
            if os.path.exists(model_path):
                self.caption_model = load_model(model_path)
                logger.info(f"Caption model loaded from {model_path}")
            else:
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Load the tokenizer
            if os.path.exists(tokenizer_path):
                with open(tokenizer_path, 'rb') as f:
                    self.tokenizer = pickle.load(f)
                logger.info(f"Tokenizer loaded from {tokenizer_path}")
            else:
                raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")
            
            # Initialize VGG16 model for feature extraction
            base = VGG16()
            self.vgg_model = Model(inputs=base.inputs, outputs=base.layers[-2].output)
            logger.info("VGG16 feature extractor initialized")
            
        except Exception as e:
            logger.error(f"Error initializing ImageCaptionGenerator: {str(e)}")
            raise
    
    def extract_features(self, image_path):
        """
        Extract features from an image using VGG16
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            numpy.ndarray: Extracted features
        """
        image = load_img(image_path, target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape((1, *image.shape))
        image = preprocess_input(image)
        feature = self.vgg_model.predict(image, verbose=0)
        return feature
    
    def idx_to_word(self, integer):
        """
        Convert integer index to word using tokenizer
        
        Args:
            integer (int): Index to convert
            
        Returns:
            str: Corresponding word or None if not found
        """
        for word, idx in self.tokenizer.word_index.items():
            if idx == integer:
                return word
        return None
    
    def predict_caption(self, image_features):
        """
        Generate caption for given image features
        
        Args:
            image_features (numpy.ndarray): Features extracted from image
            
        Returns:
            str: Generated caption
        """
        in_text = 'startseq'
        for _ in range(self.max_length):
            seq = self.tokenizer.texts_to_sequences([in_text])[0]
            seq = pad_sequences([seq], self.max_length, padding='post')
            yhat = self.caption_model.predict([image_features, seq], verbose=0)
            yhat = np.argmax(yhat)
            word = self.idx_to_word(yhat)
            if word is None: break
            in_text += ' ' + word
            if word == 'endseq':
                break
        return in_text

    def generate_caption(self, image_path, display_image=True):
        """
        Generate a cleaned caption and optionally display the image.
        """
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"No such image: {image_path}")
            # verify image integrity
            with Image.open(image_path) as img:
                img.verify()
            features = self.extract_features(image_path)
            raw = self.predict_caption(features)
            caption = raw.replace('startseq', '').replace('endseq', '').strip()
            if caption:
                caption = caption.capitalize()
                if not caption.endswith('.'):
                    caption += '.'
            logger.info(f"Caption: {caption}")
            if display_image:
                img = Image.open(image_path)
                plt.figure(figsize=(6,6))
                plt.imshow(img)
                plt.axis('off')
                plt.title(caption, pad=10)
                plt.show()
            return caption
        except Exception as e:
            logger.error(f"Caption generation failed: {e}")
            return None

    def is_model_loaded(self):
        return all([self.caption_model, self.tokenizer, self.vgg_model])
