import hashlib
import json
from pathlib import Path
import re

from huggingface_hub import hf_hub_download


DATASET = 'Hellisotherpeople/Lipogram-e'
REVISION = '5e3ddde521c24727a134e4825d2927de25784c41'
DATA_DIR = Path(__file__).resolve().parent


def extract_chapters(text):
    match = re.search(r'(?ms)^GADSBY\s*\n(.*?)^FINIS GADSBY[ \t]*$', text)
    if match is None:
        raise ValueError('Gadsby book boundaries were not found')
    parts = re.split(r'(?m)^([IVXLCDM]+)[ \t]*$', match[1])[1:]
    chapters = []
    for title, body in zip(parts[::2], parts[1::2]):
        paragraphs = [' '.join(p.split()) for p in re.split(r'\n\s*\n', body) if p.strip()]
        content = '\n\n'.join(paragraphs)
        if not content or 'e' in content.lower():
            raise ValueError(f'Chapter {title} is empty or contains e/E; inspect the source')
        chapters.append({'chapter': title, 'text': content})
    if not chapters:
        raise ValueError('No chapters were found')
    return chapters


def main(output_dir=str(DATA_DIR), val_chapters=5, revision=REVISION):
    source = Path(hf_hub_download(DATASET, 'no_e_books.txt', repo_type='dataset', revision=revision))
    raw = source.read_bytes()
    chapters = extract_chapters(raw.decode('utf-8'))
    if not 0 < val_chapters < len(chapters):
        raise ValueError('val_chapters must leave at least one training chapter')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {'dataset': DATASET, 'revision': revision, 'source_file': source.name,
                'source_sha256': hashlib.sha256(raw).hexdigest(), 'splits': {}}
    for split, rows in [('train', chapters[:-val_chapters]), ('val', chapters[-val_chapters:])]:
        path = output_dir / f'gadsby_{split}.jsonl'
        contents = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows)
        path.write_text(contents, encoding='utf-8')
        manifest['splits'][split] = {
            'chapters': [row['chapter'] for row in rows],
            'words': sum(len(row['text'].split()) for row in rows),
            'sha256': hashlib.sha256(contents.encode('utf-8')).hexdigest(),
        }
        print(f'{path}: {len(rows)} chapters, {manifest["splits"][split]["words"]} words')
    (output_dir / 'gadsby_manifest.json').write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == '__main__':
    import fire

    fire.Fire(main)
