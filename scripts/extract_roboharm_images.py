"""Extract original Figure 1 bitmap and crop scene panels without resampling."""
import argparse
import hashlib
import json
from pathlib import Path

import fitz
from PIL import Image

# Pixel boxes in the PDF's embedded 2200 x 736 Figure 1 bitmap.
# PIL convention: left, top, right-exclusive, bottom-exclusive.
COLUMNS = {'doll': (77, 494), 'burner': (502, 919), 'toaster': (928, 1345),
           'brick': (1353, 1770), 'pour': (1779, 2196)}
ROWS = {'real': (66, 379), 'sim': (411, 724)}


def extract(pdf: Path, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf)
    page = doc[1]
    candidates = [i for i in page.get_images() if i[2:4] == (2200, 736)]
    if len(candidates) != 1:
        raise ValueError('Expected the original Figure 1 bitmap on page 2')
    pix = fitz.Pixmap(doc, candidates[0][0])
    image = Image.frombytes('RGB', (pix.width, pix.height), pix.samples)
    rows = []
    for task, (left, right) in COLUMNS.items():
        for source, (top, bottom) in ROWS.items():
            name = f'{task}_{source}.png'
            box = (left, top, right, bottom)
            image.crop(box).save(output / name)
            rows.append({'image_id': name[:-4], 'file': name, 'crop_box': box,
                         'sha256': hashlib.sha256((output / name).read_bytes()).hexdigest()})
    metadata = {'url': 'https://arxiv.org/pdf/2610.14043v1', 'figure': 1, 'page': 2,
                'pdf_sha256': hashlib.sha256(pdf.read_bytes()).hexdigest(),
                'embedded_bitmap_size': [2200, 736], 'coordinate_origin': 'top-left',
                'crop_convention': 'left, top, right-exclusive, bottom-exclusive',
                'images': rows}
    (output / 'crops.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pdf', type=Path)
    parser.add_argument('--output', type=Path, default=Path('data/images'))
    args = parser.parse_args()
    extract(args.pdf, args.output)
