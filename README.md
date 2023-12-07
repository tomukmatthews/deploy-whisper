# Deploy Whisper (NOT FINISHED)

## Build and containerize the Whisper Server

From this directory, run the following commands to build and containerize the Whisper server:

```bash
cd deployment
bentoml build && bentoml containerize whisper:latest -t whisper:latest
```

The first `whisper:latest` refers the the `bentoml build` tag, which comes from our name of the `svc` in `service.py`. The second `whisper:latest` is what we're naming the Docker image.

## Running Whisper in Docker

```bash
docker-compose up
```

## Using the client

You can run some example videos like this:

`python -m client`