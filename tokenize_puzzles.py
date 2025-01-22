import pathlib

from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.readers import JsonlReader, ParquetReader
from datatrove.pipeline.tokens.tokenizer import DocumentTokenizer


if __name__ == "__main__":
    this_file = pathlib.Path(__file__)
    output_folder = this_file.parent / "data" / "tokenized"

    LocalPipelineExecutor(
        pipeline=[
            ParquetReader(
                "hf://datasets/HuggingFaceTB/finemath/finemath-4plus",  # read directly from huggingface
                glob_pattern="*.parquet",
                text_key="text",
                limit=1_000_000,  # don't need more
                read_metadata=False,
            ),
            DocumentTokenizer(
                output_folder=str((output_folder / "finemath-4plus").resolve()),
                tokenizer_name_or_path="meta-llama/Llama-3.2-3B",
                batch_size=10000,
                eos_token="<|end_of_text|>",
                shuffle=False,  # already shuffled in the reader
            ),
        ],
        tasks=10,
    ).run()

    LocalPipelineExecutor(
        pipeline=[
            JsonlReader(
                data_folder=str((this_file.parent / "data" / "raw").resolve()),
                glob_pattern="sudoku.jsonl",
                recursive=True,
            ),
            DocumentTokenizer(
                output_folder=str((output_folder / "sudoku").resolve()),
                tokenizer_name_or_path="meta-llama/Llama-3.2-3B",
                batch_size=10000,
                eos_token="<|end_of_text|>",
                shuffle=False,  # already shuffled in the reader
            ),
        ],
        tasks=1,
    ).run()
