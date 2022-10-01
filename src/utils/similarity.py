from typing import Callable, List
import clip
import numpy as np
import torch
import logging
from PIL import Image


class SimilarityComputer:
    model: torch.nn.Module
    preprocess: Callable

    def __init__(self) -> None:
        self.model, self.preprocess = clip.load("ViT-B/32")
        self.model.cuda().eval()
        input_resolution = self.model.visual.input_resolution
        context_length = self.model.context_length
        vocab_size = self.model.vocab_size

        logging.info("Model parameters:",
                     f"{np.sum([int(np.prod(p.shape)) for p in self.model.parameters()]):,}")
        logging.info("Input resolution:", input_resolution)
        logging.info("Context length:", context_length)
        logging.info("Vocab size:", vocab_size)

    def get_similarities_imgs_evidence(self, images: List[Image.Image], evidence: List[str]) -> List[float]:
        """Calculate similarities between the image and the evidence for each images.
        Uses ViT-B/32 encoder and tokenizer.

        Argument:
            images(Tuple): A pair of images which should be compared with similarities.
            evidence(str): Given text to be used to calculate the cosine similarity.

        Returns:
            Tuple[float, float]: The similarity float value of each input images
        """
        processed_images = list(map(self.preprocess, images))

        image_input = torch.tensor(np.stack(processed_images)).cuda()
        text_tokens = clip.tokenize(
            [f"It has {evidence}" for evidence in evidence]).cuda()

        with torch.no_grad():
            image_features = self.model.encode_image(image_input).float()
            text_features = self.model.encode_text(text_tokens).float()

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarities = text_features.cpu().numpy() @ image_features.cpu().numpy().T

        return [float(similarity_value) for similarity_value in similarities[0]]
