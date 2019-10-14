## CPPN

Implementation of CPPN using Chainer

### Dependencies

See `requirements.txt`

### How to init

```bash
git submodule init
git submodule update
pip install -r requirements.txt
```

For gpu support, install [CuPy](https://docs-cupy.chainer.org/en/latest/install.html).

### How to test

```
PYTHONPATH=. pytest test
```

### Format code for commit

```bash
autoflake --in-place --remove-all-unused-imports `foo.py`
isort  -y `foo.py`
yapf --in-place --style='{column_limit: 120}' `foo.py`
```

### Basic usage

TBD
