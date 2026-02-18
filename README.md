# ChaCha-Server: Hololens VQA Server - Visual Chart Question Answering Server for Hololens 2 🪇😎 
This a basic docker-based plug-and-play server setup for inference and evaluation on a Visual Language Model from Hugging Face. It is especially useful for sending requests from an Hololens App where the user queries about a chart image from their lenses. It creates another big opportunity for people with visual disabilities who want to analyze a chart image in front of them. As an example Hololens App you may refer to my Unity Hololens VQA App: <link>

## Features
- Docker-based: After setup just run `docker compose up` and use your cha-cha server from any platform!
- Fast and reliable thanks to FastApi 🐍
- Currently configured to run on cpu

## Preconditions
- Use Case 1: Only inference + Hololens 2
    - Backend must be reachable via HTTPS as Hololens 2 enforces secure networking policies. So a HTTPS reverse proxy is needed. The reverse proxy terminates TLS (handles the certificate), and forwards traffic to the internal HTTP services.
        - For my server I used: Caddy (https://caddyserver.com/docs/)
    - at least 16 GB CPU RAM
- Use Case 2: Inference and evaluation + Hololens 2
- Use Case 3: Without Hololens 2, only for Chart Question Answering

## How to setup a reverse proxy for cha-cha server

- In order to connect the server to the Hololens device you need to setup a reverse proxy server that terminates tls (e.g. Caddy) and connects to 
the server via an ssh tunnel