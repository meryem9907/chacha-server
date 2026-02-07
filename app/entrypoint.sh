#!/bin/bash

mkdir -p /data/app.db

# If you run as non-root, chmod works; chown might not (that's fine)
chmod 775 /data 
chmod 664 /data/app.db 

# start server
fastapi run ./main.py --proxy-headers --port 5000