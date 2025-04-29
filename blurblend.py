from PIL import Image, ImageFilter, ImageDraw
from typing import Optional
import logging
import numpy as np
import time


# configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import torch separately to avoid the class registration error
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch not available, using CPU fallback mode")
    TORCH_AVAILABLE = False

class BlurBlend:
    def __init__(self, model_path: str):
        """Initializing the model"""
        self.model_path = model_path
        self.device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
        self.model = None
        self.fallback_mode = False
        self.load_model()

    def load_model(self) -> None:
        """Loading the model with improved error handling"""
        try:
            if not TORCH_AVAILABLE:
                logger.warning("PyTorch not available, using fallback mode")
                self.fallback_mode = True
                return
                
            logger.info(f"Attempting to load model {self.model_path} on {self.device}")
            
            # Delay imports to avoid issues
            try:
                from transformers import pipeline
            except ImportError as e:
                logger.error(f"Failed to import transformers: {e}")
                self.fallback_mode = True
                return
            
            # Add a retry mechanism
            max_retries = 3
            current_try = 0
            
            while current_try < max_retries:
                try:
                    # Normal loading attempt
                    self.model = pipeline("image-segmentation", model=self.model_path, device=self.device)
                    logger.info(f"Model loaded successfully on {self.device}")
                    return
                except Exception as e:
                    current_try += 1
                    error_str = str(e)
                    logger.warning(f"Attempt {current_try} failed: {error_str}")
                    
                    # Check for specific errors
                    if "no running event loop" in error_str or "_path" in error_str:
                        logger.warning("PyTorch class registration issue detected, using fallback mode")
                        self.fallback_mode = True
                        return
                    
                    if current_try < max_retries:
                        time.sleep(2)  # Wait before retrying
                    else:
                        # If we're on CUDA but failing, try CPU
                        if self.device != "cpu":
                            logger.warning(f"Trying CPU instead of {self.device}")
                            self.device = "cpu"
                            try:
                                self.model = pipeline("image-segmentation", model=self.model_path, device=self.device)
                                logger.info(f"Model loaded successfully on CPU")
                                return
                            except Exception as cpu_error:
                                logger.error(f"CPU loading failed: {cpu_error}")
                        
                        # All attempts failed, use fallback mode
                        logger.warning("All loading attempts failed, using fallback mode")
                        self.fallback_mode = True
                        return
                        
        except Exception as e:
            logger.error(f"Error in load_model: {e}")
            self.fallback_mode = True
            return

    def segment_image(self, image: Image.Image) -> np.ndarray:
        """
        Segment the image to separate foreground from background
        
        Args:
            image: PIL Image to segment
            
        Returns:
            Binary mask where 1 represents foreground and 0 represents background
        """
        try:
            # If in fallback mode or no model was loaded, use simple edge detection
            if self.fallback_mode or self.model is None:
                logger.info("Using fallback segmentation method")
                return self.simple_edge_based_segmentation(image)
                
            # Ensure image is not too large for the model
            orig_size = image.size
            if max(orig_size) > 1024:
                scale = 1024 / max(orig_size)
                resized_image = image.resize(
                    (int(orig_size[0] * scale), int(orig_size[1] * scale)), 
                    Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS
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
                    mask_img = Image.fromarray(mask * 255).resize(
                        orig_size, 
                        Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS
                    )
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
                mask_img = Image.fromarray(mask * 255).resize(
                    orig_size, 
                    Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS
                )
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
                
                mask_img = Image.fromarray(mask * 255).resize(
                    orig_size, 
                    Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS
                )
                return np.array(mask_img) > 128

            logger.warning("No suitable foreground object detected in the image")
            return self.simple_edge_based_segmentation(image)
        
        except Exception as e:
            logger.error(f"Error during image segmentation: {e}")
            return self.simple_edge_based_segmentation(image)

    def simple_edge_based_segmentation(self, image: Image.Image) -> np.ndarray:
        """
        Fallback segmentation using edge detection and center focus
        
        Args:
            image: PIL Image to segment
            
        Returns:
            Binary mask where 1 represents foreground and 0 represents background
        """
        try:
            # Convert to grayscale and detect edges
            gray = image.convert("L")
            
            # Apply edge detection using filters
            edges = gray.filter(ImageFilter.FIND_EDGES)
            
            # Enhance contrast to make edges more prominent
            edges = edges.point(lambda p: p * 1.5)
            
            # Create a center weighted mask (assume subject is in center)
            width, height = image.size
            center_x, center_y = width // 2, height // 2
            
            # Create gradient mask with center focus
            mask = np.zeros((height, width), dtype=np.float32)
            for y in range(height):
                for x in range(width):
                    # Distance from center (normalized)
                    dist = np.sqrt(((x - center_x) / width) ** 2 + ((y - center_y) / height) ** 2)
                    # Center weight (1 at center, fading to 0 at edges)
                    mask[y, x] = max(0, 1 - (dist * 2.2))
            
            # Combine with edge information
            edge_array = np.array(edges) / 255.0
            mask = mask * 0.7 + edge_array * 0.3
            
            # Threshold to binary
            return (mask > 0.4).astype(np.uint8)
            
        except Exception as e:
            logger.error(f"Error in fallback segmentation: {e}")
            # Return a simple elliptical mask as last resort
            width, height = image.size
            mask = np.zeros((height, width), dtype=np.uint8)
            
            # Create elliptical mask
            center_x, center_y = width // 2, height // 2
            for y in range(height):
                for x in range(width):
                    if ((x - center_x) / (width * 0.45)) ** 2 + ((y - center_y) / (height * 0.55)) ** 2 < 1:
                        mask[y, x] = 1
                        
            return mask

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
                mask_img = Image.fromarray(mask * 255).convert('L').resize(
                    image.size,
                    Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS
                )
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
            # Return original image as fallback
            try:
                return Image.open(image_path).convert("RGB")
            except:
                # Create a blank image with error message if everything fails
                img = Image.new('RGB', (800, 400), color=(255, 255, 255))
                draw = ImageDraw.Draw(img)
                draw.text((50, 150), "Error processing image", fill=(0, 0, 0))
                return img