"""Batch processing: run the single-image pipeline over many inputs.

Used by /process-batch; each item reports its own structured result so one
failure never fabricates or blocks the rest.
"""
import concurrent.futures
from pathlib import Path

from .passport_pipeline import run as run_one, Review


def run_batch(items, max_workers=2):
    """items: [{input_path, output_path, settings}]. Returns list of results."""
    def work(item):
        try:
            return {'input': item['input_path'], 'output': item['output_path'],
                    **run_one(item['input_path'], item['output_path'],
                              item.get('settings') or {})}
        except Review as e:
            return {'input': item['input_path'], 'output': item['output_path'],
                    'ok': False, 'manualReview': True, 'error': str(e),
                    'extra': getattr(e, 'extra', {})}
        except Exception as e:
            return {'input': item['input_path'], 'output': item['output_path'],
                    'ok': False, 'manualReview': False,
                    'error': f'Processing error: {str(e)[:250]}'}

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        for r in ex.map(work, items):
            results.append(r)
    return results
