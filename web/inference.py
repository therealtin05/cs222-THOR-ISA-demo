import os
import sys
from typing import Any

import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS = {
    'base': {
        'model_path': 'google/flan-t5-base',
        'model_size': 'base',
    },
    'large': {
        'model_path': 'google/flan-t5-large',
        'model_size': 'large',
    },
}


class Config:
    def __init__(self, data: dict):
        for key, value in data.items():
            setattr(self, key, value)


if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.model import LLMBackbone
from src.utils import (
    prompt_direct_inferring,
    prompt_direct_inferring_no_target,
    prompt_for_aspect_inferring,
    prompt_for_opinion_inferring,
    prompt_for_polarity_inferring,
    prompt_for_polarity_label,
    prompt_for_target_discovery,
)

_engines: dict[str, tuple[LLMBackbone, Config]] = {}
def _config_for_model(model_id: str) -> Config:
    if model_id not in MODELS:
        raise ValueError(f'Unknown model_id "{model_id}". Choose: {list(MODELS)}')
    config_path = os.path.join(ROOT, 'config', 'config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    data['model_path'] = MODELS[model_id]['model_path']
    data['model_size'] = MODELS[model_id]['model_size']
    config = Config(data)
    config.cuda_index = 0
    config.device = torch.device(
        f'cuda:{config.cuda_index}' if torch.cuda.is_available() else 'cpu'
    )
    return config


def get_engine(model_id: str = 'base') -> tuple[LLMBackbone, Config]:
    if model_id not in _engines:
        config = _config_for_model(model_id)
        model = LLMBackbone(config=config).to(config.device)
        model.eval()
        _engines[model_id] = (model, config)
    return _engines[model_id]


def preload_models(model_ids: list[str] | None = None) -> None:
    for mid in model_ids or list(MODELS):
        get_engine(mid)


def _truncate(text: str, max_length: int = 275) -> str:
    return ' '.join(text.split()[:max_length])


def _generate(model: LLMBackbone, config: Config, prompt: str) -> str:
    batch = model.tokenizer.batch_encode_plus(
        [prompt], padding=True, return_tensors='pt', max_length=config.max_length
    )
    input_ids = batch['input_ids'].to(config.device)
    input_masks = batch['attention_mask'].to(config.device)
    outputs = model.generate(input_ids=input_ids, input_masks=input_masks)
    return outputs[0] if outputs else ''


def _label_from_text(model: LLMBackbone, text: str) -> str:
    cleaned = text.replace('<pad>', '').replace('</s>', '').strip().lower()
    for label in model.config.label_list:
        if label in cleaned:
            return label
    return model.config.label_list[0]


def target_matches_gold(gold_target: str, inferred: str) -> bool:
    """Fuzzy match for eval when the model discovers the aspect on its own."""
    g = gold_target.lower().strip()
    i = inferred.lower().strip()
    if not g or not i:
        return False
    return g in i or i in g or any(w in i for w in g.split() if len(w) > 3)


def predict_thor(text: str, target: str, model_id: str = 'base') -> dict[str, Any]:
    model, config = get_engine(model_id)
    text = _truncate(text)
    steps = []

    context_a, prompt1 = prompt_for_aspect_inferring(text, target)
    aspect_expr = _generate(model, config, prompt1)
    steps.append({'name': 'aspect', 'prompt': prompt1, 'output': aspect_expr})

    context_b, prompt2 = prompt_for_opinion_inferring(context_a, target, aspect_expr)
    opinion_expr = _generate(model, config, prompt2)
    steps.append({'name': 'opinion', 'prompt': prompt2, 'output': opinion_expr})

    context_c, prompt3 = prompt_for_polarity_inferring(context_b, target, opinion_expr)
    polarity_expr = _generate(model, config, prompt3)
    steps.append({'name': 'polarity_reason', 'prompt': prompt3, 'output': polarity_expr})

    prompt4 = prompt_for_polarity_label(context_c, polarity_expr)
    label_out = _generate(model, config, prompt4)
    polarity = _label_from_text(model, label_out)
    steps.append({'name': 'label', 'prompt': prompt4, 'output': label_out})

    result = {
        'mode': 'thor',
        'setup': 'given_target',
        'model_id': model_id,
        'target_used': target,
        'polarity': polarity,
        'steps': steps,
    }
    return result


def predict_thor_no_target(text: str, model_id: str = 'base') -> dict[str, Any]:
    """Paper-style: coarse target t is unknown; discover it then run THOR hops."""
    model, config = get_engine(model_id)
    text = _truncate(text)
    steps = []

    context0, prompt0 = prompt_for_target_discovery(text)
    inferred_target = _generate(model, config, prompt0)
    steps.append({'name': 'target_discovery', 'prompt': prompt0, 'output': inferred_target})

    context_a, prompt1 = prompt_for_aspect_inferring(text, inferred_target)
    aspect_expr = _generate(model, config, prompt1)
    steps.append({'name': 'aspect', 'prompt': prompt1, 'output': aspect_expr})

    context_b, prompt2 = prompt_for_opinion_inferring(context_a, inferred_target, aspect_expr)
    opinion_expr = _generate(model, config, prompt2)
    steps.append({'name': 'opinion', 'prompt': prompt2, 'output': opinion_expr})

    context_c, prompt3 = prompt_for_polarity_inferring(context_b, inferred_target, opinion_expr)
    polarity_expr = _generate(model, config, prompt3)
    steps.append({'name': 'polarity_reason', 'prompt': prompt3, 'output': polarity_expr})

    prompt4 = prompt_for_polarity_label(context_c, polarity_expr)
    label_out = _generate(model, config, prompt4)
    polarity = _label_from_text(model, label_out)
    steps.append({'name': 'label', 'prompt': prompt4, 'output': label_out})

    return {
        'mode': 'thor',
        'setup': 'no_target',
        'model_id': model_id,
        'inferred_target': inferred_target,
        'target_used': inferred_target,
        'polarity': polarity,
        'steps': steps,
    }


def predict_prompt(text: str, target: str, model_id: str = 'base') -> dict[str, Any]:
    model, config = get_engine(model_id)
    text = _truncate(text)
    _, prompt = prompt_direct_inferring(text, target)
    raw = _generate(model, config, prompt)
    polarity = _label_from_text(model, raw)
    return {
        'mode': 'prompt',
        'setup': 'given_target',
        'model_id': model_id,
        'target_used': target,
        'polarity': polarity,
        'steps': [{'name': 'direct', 'prompt': prompt, 'output': raw}],
    }


def predict_prompt_no_target(text: str, model_id: str = 'base') -> dict[str, Any]:
    model, config = get_engine(model_id)
    text = _truncate(text)
    _, prompt = prompt_direct_inferring_no_target(text)
    raw = _generate(model, config, prompt)
    polarity = _label_from_text(model, raw)
    return {
        'mode': 'prompt',
        'setup': 'no_target',
        'model_id': model_id,
        'polarity': polarity,
        'steps': [{'name': 'direct_no_target', 'prompt': prompt, 'output': raw}],
    }


def predict_polarity(
    text: str,
    target: str,
    mode: str,
    model_id: str = 'base',
    setup: str = 'given_target',
) -> str:
    result = predict(text, target, mode, model_id, setup=setup)
    return result['polarity']


def resolve_setup(target: str, setup: str = 'given_target') -> str:
    if not target.strip():
        return 'no_target'
    return setup if setup == 'no_target' else 'given_target'


def predict(
    text: str,
    target: str = '',
    mode: str = 'thor',
    model_id: str = 'base',
    setup: str = 'given_target',
) -> dict[str, Any]:
    if not text.strip():
        raise ValueError('Sentence text is required.')
    setup = resolve_setup(target, setup)
    if setup == 'no_target':
        if mode == 'prompt':
            return predict_prompt_no_target(text, model_id)
        if mode == 'thor':
            return predict_thor_no_target(text, model_id)
        raise ValueError(f'Unknown mode: {mode}. Use "thor" or "prompt".')
    if not target.strip():
        raise ValueError('Target aspect is required for given_target setup.')
    if mode == 'prompt':
        return predict_prompt(text, target, model_id)
    if mode == 'thor':
        return predict_thor(text, target, model_id)
    raise ValueError(f'Unknown mode: {mode}. Use "thor" or "prompt".')


def health() -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    device = 'cpu'
    if cuda_available:
        device = f'cuda:0 ({torch.cuda.get_device_name(0)})'

    models_info = {}
    for mid, spec in MODELS.items():
        loaded = mid in _engines
        models_info[mid] = {
            'model_path': spec['model_path'],
            'loaded': loaded,
        }

    vram_mb = None
    if cuda_available and _engines:
        vram_mb = round(torch.cuda.memory_allocated() / 1024**2, 1)

    return {
        'models': models_info,
        'available_model_ids': list(MODELS),
        'cuda_available': cuda_available,
        'device': device,
        'vram_allocated_mb': vram_mb,
    }
