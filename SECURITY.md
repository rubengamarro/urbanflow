# Security

This is a local portfolio demonstration with localhost Compose bindings. Remote deployments need authentication, authorization, request limits, TLS termination, secret management and network controls.

Load only trusted pipeline-generated joblib/cloudpickle models: deserialization can execute code. Checksums detect corruption but cannot authenticate an unknown artifact author. The API does not accept uploaded models or client-provided model paths.

Do not attach credentials or private data to public issues. Contact the maintainer privately using a contact channel published on their GitHub profile for sensitive reports; no reporting email or hosted security process is claimed here.
