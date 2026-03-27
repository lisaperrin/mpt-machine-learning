#!/usr/bin/env python3

import logging

from api import create_app

logging.basicConfig(level=logging.INFO)

app = create_app()

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
