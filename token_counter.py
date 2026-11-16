"""Count the exact assembled text with a fixed, documented baseline encoding."""
from functools import lru_cache
import os
from pathlib import Path

import tiktoken

ENCODING_NAME = 'cl100k_base'


@lru_cache(maxsize=1)
def get_tokenizer():
    # A project-local cache survives temporary-directory cleanup. No API is called.
    os.environ.setdefault('TIKTOKEN_CACHE_DIR', str(Path(__file__).parent / '.cache' / 'tiktoken'))
    try:
        return tiktoken.get_encoding(ENCODING_NAME)
    except Exception as error:
        raise RuntimeError('Could not load cl100k_base. Prepare its tokenizer cache using the README setup command.') from error


def count_tokens(text):
    # Source/query text resembling special tokens must remain ordinary input text.
    return len(get_tokenizer().encode(text, disallowed_special=()))
