import numpy as np

from src.utils import _get_progress, get_textgrid, memory


def encode_chunks(model, chunks, n_context=0, verbose=False):
    features = []
    n_chunks = len(chunks)
    with _get_progress(transient=True) as progress:
        if verbose:
            task = progress.add_task("Encoding chunks", total=n_chunks)

        for i in range(n_chunks):
            # Add previous chunks for context
            text = " ".join(
                [chunks[k] for k in range(max(0, i - n_context), i + 1)]
            ).strip()
            features.append(model.encode_text(text).detach().cpu().numpy().flatten())

            if verbose:
                progress.update(task, advance=1)

    return np.array(features)


# @memory.cache(ignore=["verbose"])
# TODO: Create function to instantiate model from name for use with cache
def prepare_latents(model, textgrid_path, tr, n_context=0, verbose=False):
    goodtranscript = get_textgrid(textgrid_path)
    offsets = np.array([float(x[1]) for x in goodtranscript])
    words = np.array([x[2].strip("{}").strip().lower() for x in goodtranscript])

    group_indices = np.array(offsets // tr, dtype=int)
    unique_indices = np.arange(group_indices.max())
    chunks = [" ".join(list(words[group_indices == idx])) for idx in unique_indices]

    aggregated_chunks = [
        " ".join([chunks[k] for k in range(max(0, i - n_context), i + 1)]).strip()
        for i in range(len(chunks))
    ]

    features = encode_chunks(
        model=model, chunks=chunks, n_context=n_context, verbose=verbose
    )

    return features, aggregated_chunks
