import logging

from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)


def download_llava_instruct_150k():
    hf_hub_download(
        repo_id="liuhaotian/LLaVA-Instruct-150K",
        filename="complex_reasoning_77k.json",
        repo_type="dataset",
    )

    hf_hub_download(
        repo_id="liuhaotian/LLaVA-Instruct-150K",
        filename="conversation_58k.json",
        repo_type="dataset",
    )

    logger.info("Successfully downloaded LLaVA-Instruct-150K dataset files.")
    logger.info("Files downloaded: complex_reasoning_77k.json, conversation_58k.json")
    logger.info("Make sure to download CoCoNut dataset separately, specifically the 2014 split.")


if __name__ == "__main__":
    download_llava_instruct_150k()
