# Security and Privacy

## Threat Model

This is a local research/demo application, not a public multi-user service. Security controls are therefore deliberately proportional to the risk.

## Audio Privacy

- File inference is local.
- Continuous microphone inference is local.
- The live CLI does not write captured microphone buffers to disk.
- No external inference API is used.
- No telemetry is implemented.
- Do not record conversations or identifiable speech without appropriate permission.

## UI Exposure

The Gradio demo defaults to:

```text
server_name = 127.0.0.1
share = False
```

This intentionally avoids public network exposure.

If a future deployment is made public, add:
- authentication,
- file-size limits,
- rate limits,
- reverse-proxy TLS,
- stricter file validation,
- storage/retention rules.

Those controls are not required for the current local undergraduate scope.

## Secrets

No secret or API key is required by the current project. Runtime paths, devices, and experiment settings are supplied through YAML configuration files and command-line arguments. A local `.env` file is therefore not part of the current design; `.env` remains ignored in case a future optional integration introduces secrets.

## Model/Checkpoint Trust

PyTorch checkpoint loading uses `torch.load`. Load only checkpoints created by this project or another trusted source. Arbitrary untrusted serialized model files should not be accepted from users.

## Dataset Handling

UrbanSound8K is not redistributed. Users must obtain it according to its source terms. Generated manifests contain paths and labels, not duplicated dataset audio.
