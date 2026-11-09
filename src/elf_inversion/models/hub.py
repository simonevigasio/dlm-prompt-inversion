"""Hugging Face downloads pinned to a revision, shared by every model family."""


def from_hub(load, offline):
    """``load(local_files_only)`` for True/False; "auto" reads the cache and downloads only on a miss.

    Downloads are still pinned to the configured revisions.
    """
    if offline not in (True, False, "auto"):
        raise ValueError('local_files_only must be true, false or "auto"')
    if offline != "auto":
        return load(offline)
    try:
        return load(True)
    except OSError:  # huggingface_hub's LocalEntryNotFoundError and transformers' missing files
        return load(False)
