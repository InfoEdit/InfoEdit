# InfoEdit

**Probing Global Layout Reasoning in Infographic Editing**

Cheng Yang<sup>1,\*</sup>, Chufan Shi<sup>2,\*</sup>, Huijuan Wang<sup>2,\*</sup>, Bo Shui<sup>3</sup>, Yaokang Wu<sup>4</sup>, Muzi Tao<sup>2</sup>, Yibo Yan<sup>2</sup>, Xuezhe Ma<sup>2</sup>, Taylor Berg-Kirkpatrick<sup>1</sup>

<sup>1</sup>UC San Diego &nbsp; <sup>2</sup>USC &nbsp; <sup>3</sup>UIUC &nbsp; <sup>4</sup>CMU &nbsp;&nbsp; <sup>\*</sup>Equal contribution

[Project page](https://infoedit.github.io) · [Dataset](https://huggingface.co/datasets/InfoEdit/InfoEdit) · Paper (soon)

---

Editing an infographic is not like editing a photo. Infographics encode information
through **logical relations**, so changing one element forces the surrounding elements to
adapt. We call this global layout reasoning ability **reflow**, and InfoEdit measures it.

**1,000 infographics** · **8 logical-relation families** · **4,000 editing instructions** · **4 tasks**

Across eight frontier editors, the best pixel-level model reaches **62.0%** average success
rate and the best code-level system **61.6%**; most editors fall below **7%**.

## What you need first

* **Python 3.10+**
* **A Google Cloud project with billing enabled.** Editing and judging both run as
  [Vertex AI batch prediction](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/batch-prediction)
  jobs — roughly half the price of online calls, but they do require a real GCP project.
* **The `gcloud` CLI** ([install](https://cloud.google.com/sdk/docs/install)).

Optional, per backend:

| you want to run | you also need |
|---|---|
| code-level on HTML | `playwright install chromium` |
| code-level on PPT | LibreOffice **and** poppler |
| GPT-Image-2 | `OPENAI_API_KEY` |
| Seedream | `ARK_API_KEY` (Volcengine) |
| Qwen / Hunyuan | a CUDA GPU box; see `baselines/pixel/{qwen,hunyuan}/run.sh` |

## Install

```bash
git clone https://github.com/InfoEdit/InfoEdit.git && cd InfoEdit
pip install -r requirements.txt
playwright install chromium        # HTML code-level path renders with Chromium

# only for the PPT code-level path:
#   macOS          brew install --cask libreoffice poppler
#   Debian/Ubuntu  apt-get install libreoffice poppler-utils
```

## Configure Google Cloud

```bash
export GCP_PROJECT=your-project-id

# 1. authenticate (this is what the Python SDK uses)
gcloud auth application-default login
gcloud auth application-default set-quota-project $GCP_PROJECT

# 2. enable the APIs
gcloud services enable aiplatform.googleapis.com storage.googleapis.com --project $GCP_PROJECT

# 3. create the staging bucket batch jobs read and write through.
#    It must be a SINGLE region (us-central1), not the multi-region "us".
gcloud storage buckets create gs://${GCP_PROJECT}-batch-io --location=us-central1

# 4. record it for the run scripts
cp env.example.sh env.sh     # set GCP_PROJECT inside
source env.sh
```

## Get the benchmark

```bash
huggingface-cli download InfoEdit/InfoEdit --repo-type dataset --local-dir data
```

## Run it

```bash
bash quickstart.sh      # 5 examples, edit + score, end to end
```

Batch jobs sit in a queue before they start, so even five examples usually take a few
minutes — a run that prints `JOB_STATE_QUEUED` for a while is working, not stuck.

Then the real thing, two steps per model:

```bash
# 1. produce edits
MODEL=gemini-2.5-flash-image TASK=add bash run_edit.sh

# 2. score them (Edit Compliance, Content Preservation, Success Rate)
MODEL=gemini-2.5-flash-image TASK=add bash run_eval.sh
```

Edits land in `edited_<source>_infographics[_code]/<version>/<model>/<task>/`, per-example
judgements in `eval_results/<model>/`, and `run_eval.sh` prints the EC / CP / SR table at
the end.

Both scripts read the same environment variables:

| var | values | default |
|---|---|---|
| `MODEL` | editor model id, e.g. `gemini-2.5-flash-image`, `gemini-3.5-flash` | *required* |
| `TASK` | `text_expand` · `add` · `swap_inter` · `aspect_ratio` | `text_expand` |
| `SOURCE` | `html` (800) · `ppt` (200) | `html` |
| `PATHWAY` | `image` · `code` · `code_image` · `gpt` · `seedream` | `image` |
| `LIMIT` | number of examples, empty = all | all |
| `JUDGE` | judge model | `gemini-3.1-pro-preview` |

`PATHWAY=image` uses Gemini image models; `code` / `code_image` edit the HTML or PPTX
source (`code_image` also shows the model the rendered original). Task names map to the
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
    --model_path gemini-3.1-pro-preview \
    --use_batch --gcp_project "$GCP_PROJECT" \
    --gcp_location "$GCP_LOCATION" --batch_bucket_uri "$BATCH_BUCKET_URI"

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

baselines/                  editor backends, grouped by the two pathways the
                            paper compares. Each backend is a folder with an
                            edit.py, plus recover_batch.py where the backend
                            runs as a resumable batch job:

  pixel/                      edit the rendered image  (main tables)
    gemini/                     Gemini image models (default)
    gpt/                        GPT-Image-2
    seedream/                   Seedream
    qwen/                       local Qwen-Image-Edit   (run.sh, not a PATHWAY)
    hunyuan/                    local HunyuanImage      (run.sh, not a PATHWAY)
  code/                       edit the source, then re-render
    html/                       HTML source
    ppt/                        PPTX source

  recover_batch_evaluate.py   resume an interrupted judging job
  show_jobs.sh                list Vertex AI batch jobs
```

This release covers the two experiments reported in the main tables: **pixel-level**
editing (Table 2) and **code-level** editing (Table 3). The dataset itself is
distributed via HuggingFace, so no dataset-construction code is included.

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
