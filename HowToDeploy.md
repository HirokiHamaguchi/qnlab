# How to Deploy

## Set Up

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt 
```

## Build

https://packaging.python.org/en/latest/tutorials/packaging-projects/

```bash
hatch version patch
python3 -m build --sdist
tar -tf dist/qnlab-0.0.3.tar.gz # check the content
python3 -m twine upload --repository pypi dist/*
```
