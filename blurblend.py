from PIL import Image, ImageFilter
from transformers import pipeline
import torch
from typing import Optional
import logging
import numpy as np
import time

# configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BlurBlend:
    def __init__(self, model_path: str):
        """Initializing the model"""
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.load_model()

    def load_model(self) -> None:
        """Loading the model with improved error handling"""
        try:
            logger.info(f"Attempting to load model {self.model_path} on {self.device}")
            
            # Add a retry mechanism
            max_retries = 3
            current_try = 0
            
            while current_try < max_retries:
                try:
                    self.model = pipeline("image-segmentation", model=self.model_path, device=self.device)
                    logger.info(f"Model loaded successfully on {self.device}")
                    return
                except OSError as e:
                    current_try += 1
                    if current_try < max_retries:
                        logger.warning(f"Attempt {current_try} failed, retrying... Error: {str(e)}")
                        time.sleep(2)  # Wait before retrying
                    else:
                        # If we're on CUDA but failing, try falling back to CPU
                        if self.device != "cpu":
                            logger.warning(f"Failed to load model on {self.device}, trying CPU instead")
                            self.device = "cpu"
                            try:
                                self.model = pipeline("image-segmentation", model=self.model_path, device=self.device)
                                logger.info(f"Model loaded successfully on CPU")
                                return
                            except Exception as inner_e:
                                logger.error(f"Error loading model on CPU: {inner_e}")
                                raise
                        else:
                            raise
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise RuntimeError(f"Failed to load model {self.model_path}: {str(e)}")

    def segment_image(self, image: Image.Image) -> np.ndarray:
        """
        Segment the image to separate foreground from background
        
        Args:
            image: PIL Image to segment
            
        Returns:
            Binary mask where 1 represents foreground and 0 represents background
        """
        try:
            # Ensure image is not too large for the model
            orig_size = image.size
            if max(orig_size) > 1024:
                scale = 1024 / max(orig_size)
                resized_image = image.resize(
                    (int(orig_size[0] * scale), int(orig_size[1] * scale)), 
                    Image.LANCZOS
                )
            else:
                resized_image = image

            # Run segmentation on resized image
            result = self.model(resized_image)
            
            # Create an empty mask
            mask = np.zeros(resized_image.size[::-1], dtype=np.uint8)
            
            # Special handling for clothing segmentation model
            if "segformer_b2_clothes" in self.model_path:
                # For clothing model, we want to detect all clothing items as foreground
                clothing_classes = [
                    'upper-clothes', 'coat', 'pants', 'dress', 'skirt', 'shoes',
                    'hat', 'scarf', 'face', 'hair', 'socks', 'gloves', 'jumpsuit',
                    'upper clothes', 'lower clothes', 'shirt', 'pants', 't-shirt', 
                    'jacket', 'dress', 'skirt', 'tops', 'bottoms'
                ]
                
                found_clothing = False
                for segment in result:
                    if any(clothing_class in segment['label'].lower() for clothing_class in clothing_classes):
                        segment_mask = (
                            np.array(segment['mask'])
                            if isinstance(segment['mask'], Image.Image)
                            else np.array(Image.open(segment['mask']).convert('L'))
                        )
                        mask = np.maximum(mask, (segment_mask > 128).astype(np.uint8))
                        found_clothing = True
                        logger.info(f"Found clothing item in segment labeled: {segment['label']}")
                
                if found_clothing:
                    mask_img = Image.fromarray(mask * 255).resize(orig_size, Image.LANCZOS)
                    return np.array(mask_img) > 128
            
            # Look for person/people and other common foreground classes
            person_classes = [
                'person', 'people', 'human', 'foreground', 'man', 
                'woman', 'child', 'boy', 'girl', 'individual'
            ]
            
            found_person = False
            for segment in result:
                if any(person_class in segment['label'].lower() for person_class in person_classes):
                    segment_mask = (
                        np.array(segment['mask'])
                        if isinstance(segment['mask'], Image.Image)
                        else np.array(Image.open(segment['mask']).convert('L'))
                    )
                    mask = np.maximum(mask, (segment_mask > 128).astype(np.uint8))
                    found_person = True
                    logger.info(f"Found person in segment labeled: {segment['label']}")
            
            if found_person:
                mask_img = Image.fromarray(mask * 255).resize(orig_size, Image.LANCZOS)
                return np.array(mask_img) > 128
            
            if len(result) > 0:
                largest_segment = sorted(
                    result, 
                    key=lambda x: np.sum(
                        np.array(
                            x['mask'] if isinstance(x['mask'], Image.Image) else Image.open(x['mask'])
                        ) > 128
                    ), 
                    reverse=True
                )[0]
                
                logger.info(f"No person detected, using largest object: {largest_segment['label']}")
                segment_mask = (
                    np.array(largest_segment['mask'])
                    if isinstance(largest_segment['mask'], Image.Image)
                    else np.array(Image.open(largest_segment['mask']).convert('L'))
                )
                mask = (segment_mask > 128).astype(np.uint8)
                
                mask_img = Image.fromarray(mask * 255).resize(orig_size, Image.LANCZOS)
                return np.array(mask_img) > 128

            logger.warning("No suitable foreground object detected in the image")
            return np.zeros(image.size[::-1], dtype=np.uint8)
        
        except Exception as e:
            logger.error(f"Error during image segmentation: {e}")
            return np.zeros(image.size[::-1], dtype=np.uint8)

    def apply_blur(self, image: Image.Image, mask: np.ndarray, blur_radius: int = 15) -> Image.Image:
        """
        Apply blur to the background of the image based on the mask
        
        Args:
            image: Original PIL Image
            mask: Binary mask where 1 is foreground and 0 is background
            blur_radius: Strength of the blur effect
            
        Returns:
            PIL Image with blurred background
        """
        try:
            mask = mask.astype(np.uint8)
            
            if mask.shape[:2][::-1] != image.size:
                logger.info(f"Resizing mask from {mask.shape[:2][::-1]} to {image.size}")
                mask_img = Image.fromarray(mask * 255).convert('L').resize(image.size)
                mask = np.array(mask_img) > 128
            
            mask_img = Image.fromarray((mask * 255).astype(np.uint8)).convert('L')
            blurred_img = image.copy().filter(ImageFilter.GaussianBlur(blur_radius))
            result = Image.composite(image, blurred_img, mask_img)
            
            return result
        except Exception as e:
            logger.error(f"Error applying blur: {e}")
            return image

    def process_image(self, image_path: str, output_path: Optional[str] = None, blur_radius: int = 15) -> Image.Image:
        """
        Process an image to blur its background
        
        Args:
            image_path: Path to input image
            output_path: Path to save the output image (optional)
            blur_radius: Strength of the blur effect
            
        Returns:
            Processed image with blurred background
        """
        try:
            image = Image.open(image_path).convert("RGB")
            mask = self.segment_image(image)
            result = self.apply_blur(image, mask, blur_radius)
            
            if output_path:
                result.save(output_path)
                logger.info(f"Processed image saved to {output_path}")
            
            return result
        
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            raise