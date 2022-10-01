import os
import json
from collections import namedtuple
from PIL import Image
from src.utils.similarity import SimilarityComputer

EvidenceData = namedtuple("EvidenceData", ["support_img_filename", "query_img_filename",
                          "support_img_description", "query_img_description", "counterfactual_explanation"])


def similarities_stat(similarity_list):
    sim_diff_sum = 0
    count = 0
    count_minus = 0
    for support_sim, query_sim in similarity_list:
        if support_sim - query_sim < 0.05:
            print(f"{support_sim} {query_sim}")
            count += 1
        if query_sim < support_sim:
            count_minus += 1
        sim_diff_sum += query_sim - support_sim
    return (count, count_minus, sim_diff_sum)


if __name__ == "__main__":
    sc = SimilarityComputer()
    evidence_data_filenames = os.listdir(os.path.join("evidence_data"))

    similarity_list = []
    for filename in evidence_data_filenames:
        with open(os.path.join("evidence_data", filename), "r") as fp:
            evidence_data: EvidenceData = json.load(
                fp, object_hook=lambda d: EvidenceData(**d))

        support_img = Image.open(os.path.join(
            "support_images", evidence_data.support_img_filename))
        query_img = Image.open(os.path.join(
            "query_images", evidence_data.query_img_filename))

        similarity = sc.get_similarities_imgs_evidence(
            images=[support_img, query_img], evidence=evidence_data.counterfactual_explanation)
        similarity_list.append(similarity)
        with open("log.txt", "a") as log_file:
            log_file.write(
                f"{filename}: {similarity} / {evidence_data.support_img_filename} and {evidence_data.query_img_filename}\n")

    print(similarities_stat(similarity_list))
