# OracleGraph data

## Dataset format

The loader expects JPEG images and UTF-8 JSON annotations with matching stems:

```text
oraclegraph/
├── images/page_0001.jpg
└── json/page_0001.json
```

A full annotation contains:

```json
{
  "image_id": "page_0001",
  "adversarial_scenario": ["NORMAL"],
  "oracle_graph": {
    "nodes": [
      {
        "node_id": "n_main1",
        "type": "MAIN_TEXT",
        "polygon": [10, 10, 80, 10, 80, 100, 10, 100],
        "text": "示例",
        "attributes": {}
      }
    ],
    "edges": []
  }
}
```

`sdpo_vlm.data.schema.to_short` converts content nodes to:

```json
{"n":[{"id":"m1","t":"MT","box":[10,10,80,100],"txt":"示例"}],"e":[]}
```

| Full node type | Compact type |
| --- | --- |
| `MAIN_TEXT` | `MT` |
| `INTERLINEAR_ANNOTATION` | `IA` |
| `TOP_MARGINALIA` | `TM` |
| `BOTTOM_MARGINALIA` | `BM` |
| `SIDE_MARGINALIA` | `SM` |
| `CENTER_FOLD_TEXT` | `CF` |
| `RED_SEAL_STAMP` | `RS` |

Edges use `R` for `READS_AFTER`, `A` for `ANNOTATES`, and `O` for `OVERLAPS`.
Page scaffolding is excluded from the compact content graph. Coordinates are
image pixel coordinates. Bounding rectangles and text/relations survive the
content round trip; general polygon shapes, node attributes, and z-order are
not preserved by the default compact representation.

## Procedural generator

The main 5K OracleGraph corpus is procedurally generated. Filenames such as
`iso_v1_w3_p00010` identify generator workers and page counters, not book IDs.
No human annotation agreement is implied by those generated labels.

`scripts/generate_data.py` contains the generation logic for five scenarios:
normal pages, dense interlinear annotations, seal/text intrusion, visually
missing column separators, and displaced notes. Supply your own text,
background, Chinese fonts, and optional PNG seal assets:

```bash
python scripts/generate_data.py \
  --output data/oraclegraph --n-pages 5000 --workers 64 \
  --balanced --seed 0 --max-columns 6 \
  --background /path/to/background.png \
  --text /path/to/text.txt \
  --font-main /path/to/main.ttf \
  --font-annotation /path/to/annotation.ttf \
  --font-fallback /path/to/fallback.ttf \
  --seals-dir /path/to/seals
```

The template geometry is designed for an 1885×1333 background. The source text
must be long enough for the requested pages. Worker count affects source-text
offsets and RNG streams; retain it along with the seed and all input assets
when reproducing a generation run. This example creates a new corpus; exact
reconstruction of the paper corpus requires its original assets and generation
configuration. The generator exits with an error if fewer pages are produced
than requested.

## Included example

`examples/oraclegraph/` is an original, constructed one-page illustration with
four content nodes and three relations. It demonstrates file formats and
requires no external fonts or dataset access. `examples/predictions.json`
contains its exact graph as a prediction for testing the scorer; a perfect
score on this file is not a model result.

Full corpora, commercial fonts, scanned page backgrounds, seal collections,
and model checkpoints are not included in this source release.
