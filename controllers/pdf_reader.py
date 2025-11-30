from pypdf import PdfReader
from pathlib import Path
import semchunk
from transformers import AutoTokenizer

# from docling.datamodel.pipeline_options import VlmPipelineOptions
# from docling.datamodel.base_models import InputFormat
# from docling.datamodel.pipeline_options import VlmPipelineOptions
# from docling.datamodel.pipeline_options_vlm_model import ApiVlmOptions, ResponseFormat
# from docling.document_converter import DocumentConverter, PdfFormatOption
# from docling.pipeline.vlm_pipeline import VlmPipeline


# --- Not working on a remote ollama host ---
# pipeline_options = VlmPipelineOptions(
#    enable_remote_services=True,
# )
# pipeline_options.vlm_options = ApiVlmOptions(
#    url="http://10.10.10.1:11434/v1/chat/completions",
#    params={
#        "model": "ibm/granite-docling:latest",
#    },
#    prompt="OCR the full page to markdown.",
#    timeout=90,
#    scale=1.0,
#    response_format=ResponseFormat.MARKDOWN,
#    concurrency=3,
# )
#
# converter = DocumentConverter(
#    format_options={
#        InputFormat.PDF: PdfFormatOption(
#            pipeline_cls=VlmPipeline,
#            pipeline_options=pipeline_options,
#        )
#    }
# )


chunker = semchunk.chunkerify(
    AutoTokenizer.from_pretrained("isaacus/kanon-tokenizer", force_download=True),
    512,  # maximal token size for chunk
)


def read_pdf(pdf_path: Path) -> list[str]:
    reader = PdfReader(pdf_path)

    all_text: str = ""

    for page in reader.pages:
        for row in page.extract_text(exctraction_mode="layout").split("\n"):
            if len(row) < 3:
                continue
            all_text += row
            if row[-1] == "-":
                all_text += ""
            else:
                all_text += " "

    split_sent: list[str] = []
    for sent in all_text.split("."):
        if not len(sent.strip()):
            continue
        split_sent.append(sent.strip())

    # --- old(and bad) text chunker ---
    # result: list[str] = [ ". ".join(split_sent[_:_+30]) for _ in range(0, len(split_sent), 30) ]

    return [text for text in chunker(". ".join(split_sent)) if len(text) > 70]

    # return result


# def docling_read_pdf(pdf_path: Path) -> list[str]:
#    return converter.convert(pdf_path).document.export_to_markdown()


if __name__ == "__main__":

    text = read_pdf(Path("./pdf_samples/example1.pdf"))
    print(text)
    print(len(text))
