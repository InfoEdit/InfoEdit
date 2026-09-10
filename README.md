# InfoEdit

**Probing Global Layout Reasoning in Infographic Editing**

Cheng Yang<sup>1,\*</sup>, Chufan Shi<sup>2,\*</sup>, Huijuan Wang<sup>2,\*</sup>, Bo Shui<sup>3</sup>, Yaokang Wu<sup>4</sup>, Muzi Tao<sup>2</sup>, Yibo Yan<sup>2</sup>, Xuezhe Ma<sup>2</sup>, Taylor Berg-Kirkpatrick<sup>1</sup>

<sup>1</sup>UC San Diego &nbsp; <sup>2</sup>USC &nbsp; <sup>3</sup>UIUC &nbsp; <sup>4</sup>CMU &nbsp;&nbsp; <sup>\*</sup>Equal contribution

[Project page](https://infoedit.github.io) · Paper (soon) · Dataset (soon)

---

Editing an infographic is not like editing a photo. Infographics encode information
through **logical relations**, so changing one element forces the surrounding elements to
adapt. We call this global layout reasoning ability **reflow**, and InfoEdit measures it.

**1,000 infographics** · **8 logical-relation families** · **4,000 editing instructions** · **4 tasks**

Across eight frontier editors, the best pixel-level model reaches **62.0%** average success
rate and the best code-level system **61.6%**; most editors fall below **7%**.

## Quick start

```bash
pip install -r requirements.txt

cp env.example.sh env.sh    # fill in your GCP project
source env.sh

bash quickstart.sh          # edit + score 5 examples end-to-end
```

Then run the real thing — two steps, one model at a time:

```bash
# 1. produce edits
MODEL=gemini-2.5-flash-image TASK=add bash run_edit.sh

# 2. score them (Edit Compliance, Content Preservation, Success Rate)
MODEL=gemini-2.5-flash-image TASK=add bash run_eval.sh
```

Both scripts read the same environment variables:

| var | values | default |
|---|---|---|
| `MODEL` | any editor model id | *required* |
| `TASK` | `text_expand` · `add` · `swap_inter` · `aspect_ratio` | `text_expand` |
| `SOURCE` | `html` (800) · `ppt` (200) | `html` |
| `PATHWAY` | `image` · `code` · `code_image` · `gpt` · `seedream` | `image` |
| `LIMIT` | number of examples, empty = all | all |
| `JUDGE` | judge model | `gemini-3.1-pro-preview` |

Task names map to the paper as: `text_expand`→Expand-Text, `add`→Insert-Element,
`swap_inter`→Swap-Block, `aspect_ratio`→Reshape-Canvas.

## Data

The benchmark data is released via HuggingFace Datasets and is **not** tracked in git.
Download it into this directory so the layout is:

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

python summarize_eval.py eval_results/my-model
```

Edited files must keep the original stem, e.g. `edited_html_infographics/v17/my-model/add/1_edited.png`.

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
    qwen/                       local Qwen-Image-Edit
    hunyuan/                    local HunyuanImage
  code/                       edit the source, then re-render
    html/                       HTML source
    ppt/                        PPTX source

  recover_batch_evaluate.py   resume an interrupted judging job
  show_jobs.sh                list Vertex AI batch jobs
```

This release covers the two experiments reported in the main tables: **pixel-level**
editing (Table 2) and **code-level** editing (Table 3). The dataset itself is
distributed via HuggingFace, so no dataset-construction code is included.

Editing and judging run as **Vertex AI batch prediction** jobs (~50% cheaper than online
calls). You need a GCP project with `aiplatform.googleapis.com` enabled and a
single-region (`us-central1`) staging bucket — `env.example.sh` documents the setup.

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
