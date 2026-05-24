uv init .
source .venv/bin/activate
cd rag_pdf
uv run main.py

push:
git remote set-url origin git@github.com:USERNAME/REPO.git