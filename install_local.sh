poetry export -f requirements.txt --with dev --without-hashes > requirements.txt && pip install -r requirements.txt && rm requirements.txt
