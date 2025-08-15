# Farl

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/farl.svg)](https://badge.fury.io/py/farl)
[![Coverage Status](https://coveralls.io/repos/github/nafnix/farl/badge.svg?branch=master)](https://coveralls.io/github/nafnix/farl?branch=master)
[![Code Coverage](https://codecov.io/gh/nafnix/farl/branch/master/graph/badge.svg)](https://codecov.io/gh/nafnix/farl)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A powerful and flexible FastAPI rate limiting library that provides comprehensive rate limiting capabilities for your FastAPI applications.

## Features

- **Easy Integration**: Simple setup with FastAPI applications
- **Flexible Configuration**: Support for various rate limiting strategies
- **Multiple Backends**: In-memory and Redis backend support
- **Comprehensive Protection**: Request rate limiting with customizable rules
- **Monitoring**: Built-in metrics and logging capabilities
- **Dependency Injection**: FastAPI-style dependency injection support

## Installation

```bash
pip install farl
```

### Optional Dependencies

For Redis backend support:

```bash
pip install farl[redis]
```

## Quick Start

```python
from fastapi import Depends, FastAPI

from farl import (
    AsyncFarl,
    FarlError,
    farl_exceptions_handler,
    rate_limit,
)


# Using Redis backend
farl = AsyncFarl()

app = FastAPI()
app.add_exception_handler(FarlError, farl_exceptions_handler)


@app.get(
    "/",
    dependencies=[
        Depends(
            rate_limit({"amount": 1}),
        )
    ],
)
async def pre_minute_1_request():
    return {"message": "ok"}

```
