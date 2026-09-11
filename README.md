# InfoEdit

**Probing Global Layout Reasoning in Infographic Editing**

Cheng Yang<sup>1,\*</sup>, Chufan Shi<sup>2,\*</sup>, Huijuan Wang<sup>2,\*</sup>, Bo Shui<sup>3</sup>, Yaokang Wu<sup>4</sup>, Muzi Tao<sup>2</sup>, Yibo Yan<sup>2</sup>, Xuezhe Ma<sup>2</sup>, Taylor Berg-Kirkpatrick<sup>1</sup>

<sup>1</sup>UC San Diego &nbsp; <sup>2</sup>USC &nbsp; <sup>3</sup>UIUC &nbsp; <sup>4</sup>CMU &nbsp;&nbsp; <sup>\*</sup>Equal contribution

[Project page](https://infoedit.github.io) · [Dataset](https://huggingface.co/datasets/InfoEdit/InfoEdit) · Paper (soon)

---

> **Branch: `openai-proxy`.** Models are reached through an OpenAI-compatible
> gateway instead of Vertex AI, so setup is two environment variables and no GCP
> project. This covers the **code-level** experiments (Table 3); pixel-level
> editing needs a model that returns images, which the chat-completions API does
> not express — use `main` for that.

Editing an infographic is not like editing a photo. Infographics encode information
through **logical relations**, so changing one element forces the surrounding elements to
adapt. We call this global layout reasoning ability **reflow**, and InfoEdit measures it.

**1,000 infographics** · **8 logical-relation families** · **4,000 editing instructions** · **4 tasks**

Across eight frontier editors, the best pixel-level model reaches **62.0%** average success
rate and the best code-level system **61.6%**; most editors fall below **7%**.

## What you need first

* **Python 3.10+**
* **Access to an OpenAI-compatible LLM proxy** — a key and a base URL. Any gateway
  that speaks `chat.completions` works (LiteLLM, an internal gateway, even
  OpenAI itself).
* For the PPT path only: LibreOffice and poppler.

## Install

```bash
git clone -b openai-proxy https://github.com/InfoEdit/InfoEdit.git && cd InfoEdit
pip install -r requirements.txt
playwright install chromium        # HTML path renders with Chromium

# only for the PPT path:
#   macOS          brew install --cask libreoffice poppler
#   Debian/Ubuntu  apt-get install libreoffice poppler-utils
```

## Configure the proxy

```bash
cp env.example.sh env.sh    # set OPENAI_API_KEY and OPENAI_BASE_URL
source env.sh
```

Check that the model you plan to use actually works on your key — gateways often
gate models per key, and a model that lists is not necessarily a model you can call:

```bash
python llm_client.py --probe gemini-3.5-flash
```

It reports three things: **text** (needed to edit), **image input** (needed to
judge) and **JSON mode**. A model that fails image input can still be an editor,
just not the judge.

## Get the benchmark

```bash
huggingface-cli download InfoEdit/InfoEdit --repo-type dataset --local-dir data
```

Newer `huggingface_hub` releases rename that command to `hf download ...`. If neither is
on your PATH, the Python API does the same thing:

```bash
python -c "from huggingface_hub import snapshot_download; \
snapshot_download('InfoEdit/InfoEdit', repo_type='dataset', local_dir='data')"
```

## Run it

```bash
bash quickstart.sh      # 5 examples, edit + score, end to end
```

Calls go to the proxy directly, `--num_workers` at a time (default 8). Raise `WORKERS`
if your key allows more concurrency.

Then the real thing, two steps per model:

```bash
# 1. produce edits
MODEL=gemini-3.5-flash TASK=add bash run_edit.sh

# 2. score them (Edit Compliance, Content Preservation, Success Rate)
MODEL=gemini-3.5-flash TASK=add bash run_eval.sh
```

Edits land in `edited_<source>_infographics[_code]/<version>/<model>/<task>/`, per-example
judgements in `eval_results/<model>/`, and `run_eval.sh` prints the EC / CP / SR table at
the end.

Both scripts read the same environment variables:

| var | values | default |
|---|---|---|
| `MODEL` | editor model id — must match `PATHWAY` (see below) | *required* |
| `TASK` | `text_expand` · `add` · `swap_inter` · `aspect_ratio` | `text_expand` |
| `SOURCE` | `html` (800) · `ppt` (200) | `html` |
| `PATHWAY` | `code` · `code_image` | `code` |
| `LIMIT` | number of examples, empty = all | all |
| `JUDGE` | judge model | `gemini-3.1-pro-preview` |

Both pathways use a **text** model — they rewrite source, they do not draw pixels:

| PATHWAY | what it does | example MODEL |
|---|---|---|
| `code` | rewrites the HTML/PPTX source from the source alone | `gemini-3.5-flash` |
| `code_image` | same, but also shows the model the rendered original | `gemini-3.5-flash` |

`code_image` needs a model that accepts image input — check with
`python llm_client.py --probe MODEL`. Task names map to the
paper as: `text_expand`→Expand-Text, `add`→Insert-Element, `swap_inter`→Swap-Block,
`aspect_ratio`→Reshape-Canvas.

## Data

The benchmark lives on the Hugging Face Hub at
[InfoEdit/InfoEdit](https://huggingface.co/datasets/InfoEdit/InfoEdit) and is **not**
tracked in git. The download above gives you:

```
data/
├── html_infographics/v17/     800 × {.html, .png, .json}
├── ppt_infographics/v7/       200 × {.png, .json} + source .pptx
├── editing_prompts_html/      v17.<task>.jsonl   (800 each)
└── editing_prompts_ppt/       v7.<task>.jsonl    (200 each)
```

All main-experiment numbers use `prompt_index 0` of each record.

## Evaluating your own model

You do not have to use our editing scripts. Produce edited images yourself, write them to
a directory named after your model, then point the judge at it:

```bash
python evaluate_edits.py \
    --input_file data/editing_prompts_html/v17.jsonl \
    --edited_dir  edited_html_infographics/v17/my-model \
    --output_file eval_results/my-model/html_v17.jsonl \
    --operation add --detailed \
    --model_path gemini-3.5-flash --num_workers 8

python summarize_eval.py v17 --model my-model --prefix html --ops add
```

Lay the edits out as `<edited_dir>/<task>/{id}_edited_1.png` — for the command above that
is `edited_html_infographics/v17/my-model/add/1_edited_1.png`. The `{id}` must match the
`id` in the prompt record, and `_1` is the variant index.

## Repository layout

```
run_edit.sh / run_eval.sh   the two commands you normally need
quickstart.sh               end-to-end smoke test
evaluate_edits.py           the reflow-aware MLLM judge
summarize_eval.py           print EC / CP / SR from a results directory

llm_client.py               the one place the proxy is called

baselines/code/             the pathways this branch supports
    html/                     rewrites the HTML source, re-renders with Chromium
    ppt/                      rewrites the slide via python-pptx, renders via LibreOffice

baselines/pixel/            Vertex-only, unported — see the main branch
```

This branch covers **code-level** editing (Table 3). The dataset is distributed via
HuggingFace, so no dataset-construction code is included.

## License

Code is MIT ([LICENSE](LICENSE)). Benchmark data is CC BY 4.0 ([DATA_LICENSE](DATA_LICENSE)).

## Citation

```bibtex
@inproceedings{yang2026infoedit,
  title     = {InfoEdit: Probing Global Layout Reasoning in Infographic Editing},
  author    = {Yang, Cheng and Shi, Chufan and Wang, Huijuan and Shui, Bo and
               Wu, Yaokang and Tao, Muzi and Yan, Yibo and Ma, Xuezhe and
               Berg-Kirkpatrick, Taylor},
  booktitle = {Proceedings of EMNLP},
  year      = {2026}
}
```
