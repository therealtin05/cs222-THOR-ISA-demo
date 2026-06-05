import json
import os
import xml.etree.ElementTree as ET
from typing import Any

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

XML_FILES = {
    'laptops': os.path.join(
        ROOT, 'data', 'laptops', 'Laptops_Test_Gold_Implicit_Labeled.xml'
    ),
    'restaurants': os.path.join(
        ROOT, 'data', 'restaurants', 'Restaurants_Test_Gold_Implicit_Labeled.xml'
    ),
}

COMPARISON_JSON = os.path.join(
    os.path.dirname(__file__), 'data', 'mode_comparison_base.json'
)

CURATED_IDS = {
    'laptops': {'958:1', '282:9', '314:21', '787:270'},
    'restaurants': {
        '35390182#756337#5',
        '33071731#1007204#3',
        '33085939#758010#0',
        '35709337#1579632#6',
    },
}


def _parse_aspects(xml_path: str, domain: str, implicit_only: bool = False) -> list[dict[str, Any]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    examples = []

    for sentence in root.findall('sentence'):
        sid = sentence.get('id', '')
        text_el = sentence.find('text')
        if text_el is None or not text_el.text:
            continue
        text = text_el.text.strip()

        for aspect in sentence.findall('.//aspectTerm'):
            implicit = aspect.get('implicit_sentiment') == 'True'
            if implicit_only and not implicit:
                continue
            term = aspect.get('term', '').strip()
            polarity = aspect.get('polarity', 'neutral')
            if not term:
                continue

            examples.append({
                'id': f'{domain}:{sid}:{term}',
                'domain': domain,
                'sentence_id': sid,
                'text': text,
                'target': term,
                'gold_polarity': polarity,
                'implicit': implicit,
            })

    return examples


def load_all_eval_examples(implicit_only: bool = False) -> list[dict[str, Any]]:
    """SemEval test XML includes gold polarity on each aspectTerm."""
    all_examples = []
    for domain, path in XML_FILES.items():
        if os.path.exists(path):
            all_examples.extend(_parse_aspects(path, domain, implicit_only=implicit_only))
    return all_examples


def load_demo_examples(max_per_domain: int = 6) -> list[dict[str, Any]]:
    all_examples = []
    for domain, path in XML_FILES.items():
        if not os.path.exists(path):
            continue
        implicit = _parse_aspects(path, domain, implicit_only=True)
        curated = [e for e in implicit if e['sentence_id'] in CURATED_IDS.get(domain, set())]
        pool = curated if curated else implicit[:max_per_domain]
        for ex in pool[:max_per_domain]:
            ex = dict(ex)
            ex['category'] = 'curated'
            all_examples.append(ex)

    return all_examples[:12]


def load_comparison_cases() -> dict[str, Any]:
    if not os.path.exists(COMPARISON_JSON):
        return {
            'available': False,
            'message': 'Run: python web/compare_modes.py (see web/README.md)',
            'thor_wins': [],
            'prompt_wins': [],
        }
    with open(COMPARISON_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)
    data['available'] = True
    return data


def load_all_examples_for_ui() -> dict[str, Any]:
    curated = load_demo_examples()
    comp = load_comparison_cases()

    thor_wins = []
    for row in comp.get('thor_wins', [])[:8]:
        ex = dict(row)
        ex['category'] = 'thor_wins'
        ex['note'] = f"Gold: {row['gold_polarity']} · THOR: {row['thor_pred']} · Prompt: {row['prompt_pred']}"
        thor_wins.append(ex)

    prompt_wins = []
    for row in comp.get('prompt_wins', [])[:8]:
        ex = dict(row)
        ex['category'] = 'prompt_wins'
        ex['note'] = f"Gold: {row['gold_polarity']} · THOR: {row['thor_pred']} · Prompt: {row['prompt_pred']}"
        prompt_wins.append(ex)

    return {
        'curated': curated,
        'thor_wins': thor_wins,
        'prompt_wins': prompt_wins,
        'comparison_stats': {
            'available': comp.get('available', False),
            'thor_accuracy': comp.get('thor_accuracy'),
            'prompt_accuracy': comp.get('prompt_accuracy'),
            'thor_wins_count': comp.get('thor_wins_count'),
            'prompt_wins_count': comp.get('prompt_wins_count'),
            'total': comp.get('total'),
            'implicit_only': comp.get('implicit_only'),
        },
    }


def get_example_by_id(example_id: str) -> dict[str, Any] | None:
    bundle = load_all_examples_for_ui()
    for group in ('curated', 'thor_wins', 'prompt_wins'):
        for ex in bundle[group]:
            if ex['id'] == example_id:
                return ex
    return None
