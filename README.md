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

Everything this branch needs is two variables:

```bash
cp env.example.sh env.sh
```

Edit `env.sh`:

```bash
export OPENAI_API_KEY=sk-...                         # the key your gateway issued
export OPENAI_BASE_URL=https://your-gateway/v1       # note the /v1 suffix
```

Then load them into your shell — every command below assumes you have:

```bash
source env.sh
```

Two things that commonly go wrong:

* **`OPENAI_BASE_URL` must end in `/v1`** (or whatever prefix your gateway
  mounts the OpenAI routes under). Pointing at the bare host gives 404s on
  every call.
* **The variables have to be exported**, not just set — the Python scripts read
  them from the environment. `source env.sh` does this; running `bash env.sh`
  does not.

### Check your key before a long run

Gateways usually gate models per key, and a model that appears in `/models` is
not necessarily one you are allowed to call. Probe it first:

```bash
python llm_client.py --probe gpt-5.6-sol
```

```
proxy : OpenAI-compatible proxy at https://your-gateway/v1
model : gpt-5.6-sol

  text         ✅  'ok'
  image input  ✅  'red'  -> usable as a judge
  json mode    ✅  '{"ok": true}'
```

The three lines answer three different questions:

| line | if it fails |
|---|---|
| **text** | the model is unusable on this key — nothing else will work |
| **image input** | still fine as an *editor*, but it cannot be the *judge*, which compares the original and edited images |
| **json mode** | harmless — the judge never asks for JSON mode; it parses the reply and strips a ``` fence if one is there |

A 403 like `user not allowed to access model` means the key's tier does not
cover that model — ask whoever runs the gateway, or pick another model.

### Which model goes where

```bash
export MODEL=gpt-5.6-sol               # the editor, rewrites the source
export JUDGE=gemini-3.1-pro-preview    # scores the result, needs image input
```

`MODEL` only has to handle text (and images too, if you use `PATHWAY=code_image`).
`JUDGE` must pass the **image input** probe, since judging means looking at the
before and after renders. Probe both.

### Models used in our runs

| role | model | why |
|---|---|---|
| editor | `gpt-5.6-sol` | text model, rewrites the source |
| editor | `claude-opus-5` | text model, rewrites the source |
| judge | `gemini-3.1-pro-preview` | same judge as the paper; accepts image input |

Image models such as `gemini-2.5-flash-image` do **not** belong here — this branch
never asks a model to draw, only to rewrite source. Those live on `main`.

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
MODEL=gpt-5.6-sol TASK=add bash run_edit.sh

# 2. score them (Edit Compliance, Content Preservation, Success Rate)
MODEL=gpt-5.6-sol TASK=add bash run_eval.sh
```

`run_eval.sh` prints the EC / CP / SR table at the end. The files it worked from, for
`MODEL=gpt-5.6-sol TASK=add` on HTML:

```
edited_html_infographics_code/v17/gpt-5.6-sol_code/add/   edited source + render
eval_results/gpt-5.6-sol_code/                            per-example judgements
```

Note the `_code` suffix on both the directory and the model name — it keeps `code` and
`code_image` runs of the same model apart.

Both scripts read the same environment variables:

| var | values | default |
|---|---|---|
| `MODEL` | editor model id (a **text** model) | *required* |
| `TASK` | `text_expand` · `add` · `swap_inter` · `aspect_ratio` | `text_expand` |
| `SOURCE` | `html` (800) · `ppt` (200) | `html` |
| `PATHWAY` | `code` · `code_image` | `code` |
| `LIMIT` | number of examples, empty = all | all |
| `JUDGE` | judge model | `gemini-3.1-pro-preview` |

Both pathways use a **text** model — they rewrite source, they do not draw pixels:

| PATHWAY | what it does | example MODEL |
|---|---|---|
| `code` | rewrites the HTML/PPTX source from the source alone | `gpt-5.6-sol` |
| `code_image` | same, but also shows the model the rendered original | `claude-opus-5` |

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
    --model_path gemini-3.1-pro-preview --num_workers 8   # the judge: needs image input

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
