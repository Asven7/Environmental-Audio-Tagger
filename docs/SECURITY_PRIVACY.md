# Security and privacy

## Deployment boundary

Environmental Audio Tagger is a **local research and engineering application**, not a hardened public multi-user inference service. The Gradio UI defaults to localhost (`127.0.0.1`) with `share=False`. Do not expose it publicly without adding authentication, file-size controls, rate limiting, secure transport and data-retention safeguards.

## Audio data handling

File prediction executes locally; no remote model API or telemetry component is implemented. The local `sounddevice` CLI does not intentionally persist captured microphone chunks. Browser/UI components may have their own temporary-file and browser permission behavior, so avoid sensitive recordings and consider their retention in any deployment. Obtain appropriate consent for identifiable speech or recordings.

UrbanSound8K raw files are not distributed in the repository. Mixture manifests contain source paths, annotations and transformation recipes.

## Checkpoint trust

`src/esaudio/checkpoints.py` and some related tools use `torch.load` with `weights_only=False` in the audited revision. That flag can deserialize arbitrary Python objects from a checkpoint, so **only load checkpoints from trusted sources**. Do not accept an untrusted user-supplied `.pt` file as safe merely because it has the expected filename or extension.

The security audit recommends evaluating `weights_only=True` compatibility and restricting the checkpoint payload format in a separately reviewed code change, with regression tests. This documentation cleanup does **not** change checkpoint loading semantics.

## Repository hygiene

`data/`, local virtual environments and generated research checkpoints are ignored by Git. `.env` files are ignored as a precaution; the current application does not require external API keys. Check any files before committing them; source paths, metadata and locally collected audio can reveal private information.

The MIT code license does not replace independent licenses for the dataset and third-party dependencies. See [References](REFERENCES.md).
