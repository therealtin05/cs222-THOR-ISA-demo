# THOR-ISA Web Demo

Interactive demo for implicit sentiment analysis using **google/flan-t5-base** and **google/flan-t5-large** (both kept in GPU memory). No OpenAI API required.

## Gold labels

The SemEval test XML files include gold sentiment on each `aspectTerm` (`polarity="positive|negative|neutral"`). Implicit targets are marked `implicit_sentiment="True"`. The web UI shows gold labels for examples only (not sent to the model).

## Target given vs not given

The **released author code** (`[src/loader.py](src/loader.py)`, `[main.py](main.py)`) always uses the gold `aspectTerm` as target `t` in every prompt. That matches the paper’s SemEval setup: *given sentence X and target term t*.

What the paper does **not** give upfront is the **fine-grained aspect** `a` (hop 1 infers it, e.g. “portability” for target “new mobile phone”). Our demo previously passed the SemEval `aspectTerm` as `t`, which is already quite specific.

**Extension in this web app:** leave **Target aspect** blank to run automatically:

- **THOR (no target):** discover aspect/entity → then the usual 3 THOR hops (5 generations total)
- **Prompt (no target):** one-shot aspect + polarity question

Gold `target` from examples is still shown for reference; if the field is blank it is not used by the model (only for `target_matches_gold` when you had gold filled for eval).

```bash
python web/compare_modes.py --setup no_target   # saves web/data/mode_comparison_base_no_target.json
```

## THOR vs 1-step comparison (flan-t5-base)

Find cases where THOR matches gold but 1-step prompt does not (and vice versa) on the implicit test set:

```bash
source .venv/bin/activate
python web/compare_modes.py          # all 442 implicit aspects (~1–2 hours)
python web/compare_modes.py --limit 30   # quick sample
```

Results are saved to `web/data/mode_comparison_base.json` and appear in the example dropdown.

## Setup

```bash
cd cs222-THOR-ISA-demo
python3 -m virtualenv .venv   # use this if `python3 -m venv` fails (needs python3-venv on Ubuntu)
source .venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

First run downloads the model from Hugging Face (~1GB).

## Run

From the `THOR-ISA` directory (so `src` imports resolve):
```bash
uvicorn web.app:app --host 0.0.0.0 --port 8000 --reload
```


Open **[http://localhost:8000](http://localhost:8000)**

## API

- `GET /api/health` — device and model status
- `GET /api/examples` — curated SemEval implicit examples
- `POST /api/predict` — body: `{"text": "...", "target": "...", "mode": "thor"|"prompt", "model_id": "base"|"large"}`
- `GET /api/eval-info` — test set sizes and label field

## Modes

- **thor** — 4-step chain-of-thought (slower, shows reasoning steps)
- **prompt** — single-step direct sentiment (faster)

