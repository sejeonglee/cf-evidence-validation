import os
import unittest
from PIL import Image
from src.utils.similarity import SimilarityComputer


class TestCLIPSimilarityCompute(unittest.TestCase):
    def setUp(self) -> None:
        self.sim_computer = SimilarityComputer()
        return super().setUp()

    def test_unittestmodule(self):
        self.assertTrue(True)

    def test_similarity_returns_float_values(self):
        images = [Image.open(os.path.join("support_images", "Least_Flycatcher_0001_30221.jpg")),
                  Image.open(os.path.join("query_images", "Yellow_Bellied_Flycatcher_0020_795482.jpg"))]
        similarities = self.sim_computer.get_similarities_imgs_evidence(
            images=images, evidence="hi")
        self.assertIsInstance(similarities, list)

        for item in similarities:
            self.assertIsInstance(item, float)

    def test_different_similarity_between_different_class_with_counterfactual(self):

        test_evidence_explanation = ["beautiful yellow finch"]
        different_class_images = [Image.open(os.path.join("support_images", "Green_Kingfisher_0066_71200.jpg")),
                                  Image.open(os.path.join("query_images", "Western_Meadowlark_0001_78676.jpg"))]
        similarities = self.sim_computer.get_similarities_imgs_evidence(
            images=different_class_images, evidence=test_evidence_explanation)
        self.assertTrue(abs(similarities[0] - similarities[1]) > 0.08)
