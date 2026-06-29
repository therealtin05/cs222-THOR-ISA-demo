## CS222 Group 1: **THOR: Three-hop Reasoning for Implicit Sentiment**

## Setup

```bash
git clone https://github.com/therealtin05/cs222-THOR-ISA-demo.git
cd cs222-THOR-ISA-demo
python3 -m virtualenv .venv   # use this if `python3 -m venv` fails (needs python3-venv on Ubuntu)
source .venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```
## Run

```bash
uvicorn web.app:app --host 0.0.0.0 --port 8000 --reload
```
Then open: http://localhost:8000