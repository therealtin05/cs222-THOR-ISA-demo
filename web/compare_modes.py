#!/usr/bin/env python3
"""Compare THOR vs 1-step prompt on flan-t5-base against gold XML labels."""
import argparse
import json
import os
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tqdm import tqdm
from web.examples import load_all_eval_examples
from web.inference import get_engine, predict, target_matches_gold

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), 'data', 'mode_comparison_base.json')
OUTPUT_NO_TARGET = os.path.join(os.path.dirname(__file__), 'data', 'mode_comparison_base_no_target.json')


def run_compare(limit: int | None = None, implicit_only: bool = True, setup: str = 'given_target'):
    get_engine('base')
    examples = load_all_eval_examples(implicit_only=implicit_only)
    if limit:
        examples = examples[:limit]

    thor_correct_prompt_wrong = []
    prompt_correct_thor_wrong = []
    both_correct = both_wrong = 0
    thor_correct = prompt_correct = 0
    total = len(examples)

    desc = f'compare base ({setup}) thor vs prompt'
    for ex in tqdm(examples, desc=desc):
        gold = ex['gold_polarity']
        try:
            thor_r = predict(ex['text'], ex['target'], 'thor', 'base', setup=setup)
            prompt_r = predict(ex['text'], ex['target'], 'prompt', 'base', setup=setup)
            thor_pred = thor_r['polarity']
            prompt_pred = prompt_r['polarity']
            inferred = thor_r.get('inferred_target', '')
            aspect_match = (
                target_matches_gold(ex['target'], inferred) if setup == 'no_target' else True
            )
        except Exception as err:
            print(f"skip {ex['id']}: {err}")
            continue

        thor_ok = thor_pred == gold and aspect_match
        prompt_ok = prompt_pred == gold and aspect_match
        if thor_ok:
            thor_correct += 1
        if prompt_ok:
            prompt_correct += 1

        row = {
            **{k: ex[k] for k in ('id', 'domain', 'text', 'target', 'gold_polarity', 'implicit')},
            'thor_pred': thor_pred,
            'prompt_pred': prompt_pred,
            'setup': setup,
        }
        if setup == 'no_target':
            row['inferred_target'] = inferred
            row['target_matches_gold'] = aspect_match

        if thor_ok and prompt_ok:
            both_correct += 1
        elif not thor_ok and not prompt_ok:
            both_wrong += 1
        elif thor_ok and not prompt_ok:
            thor_correct_prompt_wrong.append(row)
        elif prompt_ok and not thor_ok:
            prompt_correct_thor_wrong.append(row)

    out_path = OUTPUT_NO_TARGET if setup == 'no_target' else OUTPUT_PATH
    result = {
        'model': 'google/flan-t5-base',
        'setup': setup,
        'created_at': datetime.utcnow().isoformat() + 'Z',
        'implicit_only': implicit_only,
        'total': total,
        'thor_accuracy': round(thor_correct / total, 4) if total else 0,
        'prompt_accuracy': round(prompt_correct / total, 4) if total else 0,
        'both_correct': both_correct,
        'both_wrong': both_wrong,
        'thor_wins_count': len(thor_correct_prompt_wrong),
        'prompt_wins_count': len(prompt_correct_thor_wrong),
        'thor_wins': thor_correct_prompt_wrong,
        'prompt_wins': prompt_correct_thor_wrong,
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Saved {out_path}")
    print(f"THOR acc: {result['thor_accuracy']:.1%}  Prompt acc: {result['prompt_accuracy']:.1%}")
    print(f"THOR correct / prompt wrong: {len(thor_correct_prompt_wrong)}")
    print(f"Prompt correct / THOR wrong: {len(prompt_correct_thor_wrong)}")
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--limit', type=int, default=None, help='Max examples (default: all)')
    p.add_argument('--all-aspects', action='store_true', help='Include explicit aspects too')
    p.add_argument(
        '--setup',
        choices=['given_target', 'no_target'],
        default='given_target',
        help='given_target=paper/repo default; no_target=infer aspect first',
    )
    args = p.parse_args()
    run_compare(
        limit=args.limit,
        implicit_only=not args.all_aspects,
        setup=args.setup,
    )
