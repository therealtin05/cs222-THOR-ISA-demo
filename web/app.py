import os
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from web.examples import get_example_by_id, load_all_examples_for_ui, load_all_eval_examples
from web.inference import MODELS, health, preload_models, predict, resolve_setup, target_matches_gold

STATIC_DIR = os.path.join(os.path.dirname(__file__), 'static')


@asynccontextmanager
async def lifespan(app: FastAPI):
    preload_models(['base', 'large'])
    yield


app = FastAPI(title='THOR-ISA Demo', lifespan=lifespan)
app.mount('/static', StaticFiles(directory=STATIC_DIR), name='static')


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1)
    target: str = ''
    mode: Literal['thor', 'prompt'] = 'thor'
    model_id: Literal['base', 'large'] = 'base'
    setup: Literal['given_target', 'no_target'] = 'given_target'


@app.get('/')
async def index():
    return FileResponse(os.path.join(STATIC_DIR, 'index.html'))


@app.get('/api/health')
async def api_health():
    return health()


@app.get('/api/models')
async def api_models():
    return {'models': MODELS}


@app.get('/api/examples')
async def api_examples():
    return load_all_examples_for_ui()


@app.get('/api/eval-info')
async def api_eval_info():
    implicit = load_all_eval_examples(implicit_only=True)
    all_aspects = load_all_eval_examples(implicit_only=False)
    return {
        'has_gold_labels': True,
        'label_field': 'aspectTerm@polarity',
        'implicit_test_aspects': len(implicit),
        'all_test_aspects': len(all_aspects),
        'domains': list({'laptops', 'restaurants'}),
    }


@app.get('/api/examples/{example_id}')
async def api_example(example_id: str):
    ex = get_example_by_id(example_id)
    if ex is None:
        raise HTTPException(status_code=404, detail='Example not found')
    return ex


@app.post('/api/predict')
async def api_predict(body: PredictRequest):
    try:
        setup = resolve_setup(body.target, body.setup)
        result = predict(
            body.text,
            body.target,
            body.mode,
            body.model_id,
            setup=setup,
        )
        if setup == 'no_target' and body.target.strip():
            result['gold_target'] = body.target.strip()
            result['target_matches_gold'] = target_matches_gold(
                body.target, result.get('inferred_target', result.get('target_used', ''))
            )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
