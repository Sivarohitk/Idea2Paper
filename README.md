# Idea2Paper – **Upload → Find Related arXiv Papers → Draft (no paid LLM)**

A small, Windows‑friendly research helper. You upload your **notes/experiment log** (PDF/DOCX/TXT/MD/CSV).  
The app extracts **keywords**, pulls **related arXiv papers**, ranks them with **SPECTER**, optionally reads **full PDFs**, then produces a **research‑style Markdown draft** using **PEGASUS‑arXiv** summarization.

> **No paid LLMs required.** Everything runs locally with Hugging Face models. GPU is optional but recommended.

---

## 🚀 Quickstart

### 1) Clone & create a virtual environment (Windows PowerShell / CMD)

```bash
git clone https://github.com/<your-username>/Idea2Paper.git
cd Idea2Paper

# Python 3.10–3.12 recommended
python -m venv .venv
.venv\Scripts\activate
```

### 2) Install dependencies

> If you have an NVIDIA GPU and want CUDA: install the **matching PyTorch** wheel *before* the rest (see notes below). Otherwise just do:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3) Run the app

```bash
streamlit run src/app.py
```

Open the printed **Local URL** in your browser.

---

## 🧭 How to use

1. **Upload** your draft/notes (PDF, DOCX, TXT, MD, CSV).  
2. Click **“Extract text & suggest keywords.”**  
3. Click **“Retrieve from arXiv.”** (tweak keywords if needed).  
4. Click **“Rank with SPECTER.”**  
5. Optionally check **“Use full PDFs for background”** (slower but better).  
6. Click **“Summarize & Generate.”** → preview + download Markdown draft (saved in `drafts/`).

---

## 🧩 Pipeline

```mermaid
flowchart TD
    A[Upload notes / draft] --> B[Keyword Extraction]
    B --> C[arXiv Retrieval]
    C --> D[SPECTER Ranking]
    D --> E[Full‑PDF Extraction (optional)]
    D -. fallback .-> F[Abstracts Only]
    E --> G[PEGASUS Summarization]
    F --> G
    G --> H[Markdown Draft]
    style A fill:#E8F0FE,stroke:#3367D6,stroke-width:2px
    style B fill:#E6F4EA,stroke:#34A853,stroke-width:2px
    style C fill:#E6F4EA,stroke:#34A853,stroke-width:2px
    style D fill:#E0F7FA,stroke:#00ACC1,stroke-width:2px
    style E fill:#FFF7E6,stroke:#FBBC04,stroke-width:2px
    style F fill:#FFF7E6,stroke:#FBBC04,stroke-width:2px,stroke-dasharray: 4 2
    style G fill:#F3E5F5,stroke:#8E24AA,stroke-width:2px
    style H fill:#F1F8E9,stroke:#7CB342,stroke-width:2px
```

---

## 📁 Repo layout

```
Idea2Paper/
├─ src/
│  ├─ app.py              # Streamlit UI
│  ├─ config.py           # central configuration
│  ├─ ingest.py           # parse PDFs/DOCX/TXT/MD/CSV
│  ├─ keywords.py         # RAKE/KeyBERT-style extraction
│  ├─ retrieval.py        # arXiv API queries
│  ├─ ranker.py           # SPECTER ranking (SentenceTransformers)
│  ├─ summarizer.py       # PEGASUS-arXiv summarization (CPU/GPU)
│  ├─ generator.py        # compose Markdown draft
│  └─ utils.py            # helpers
├─ drafts/                # generated .md drafts (git-ignored)
├─ data_samples/          # (optional) sample uploads
├─ flowchart/pipeline.md  # same Mermaid diagram
├─ notebook/demo.ipynb    # minimal, reproducible demo
├─ requirements.txt
└─ .gitignore
```

---

## ⚙️ Configuration

You can override defaults via environment variables (or `.env`). Useful ones:

| Variable | Default | Meaning |
|---|---|---|
| `IDEA2PAPER_DEVICE` | auto | Force `cpu` or `cuda` |
| `TOP_K` | 10 | Top ranked papers to use |
| `MAX_ARXIV_RESULTS` | 40 | Raw results before ranking |
| `USE_FULL_PDFS` | `true` | Download & extract PDFs for background |
| `SUM_MODEL` | `google/pegasus-arxiv` | HF seq2seq model |
| `EMBED_MODEL` | `sentence-transformers/allenai-specter` | Paper embedder |
| `PDF_TIMEOUT` | 30 | Seconds per PDF |
| `PDF_MAX_BYTES` | 50_000_000 | Size cap per PDF |
| `PDF_CHAR_CAP` | 150_000 | Truncate very long texts |

Example (PowerShell):

```powershell
setx TOP_K 8
setx USE_FULL_PDFS false
```

Reopen your shell to pick up `setx` values.

---

## ⚡ GPU notes (CUDA)

- This project works on **CPU**. For speed, use **CUDA**:
  1. Install a CUDA‑enabled PyTorch build compatible with your driver (e.g. cu121):  
     ```bash
     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
     ```
  2. Then install the rest of the requirements:
     ```bash
     pip install -r requirements.txt
     ```
- If you hit a warning about **`torch.load` vulnerability (CVE‑2025‑32434)**, upgrade PyTorch to a safe version or prefer models that ship **safetensors**. The code already requests safetensors where possible.

Check your setup inside Python:
```python
import torch; print(torch.__version__, torch.cuda.is_available())
```

---

## ✅ What works well

- Fast **SPECTER** ranking – surfaces relevant arXiv papers.
- **Full‑PDF** background (when accessible) improves coherence.
- Reproducible, no paid API; runs fully local.

## ⚠️ What disappointed me (current limits)

- **Abstractiveness**: PEGASUS‑arXiv can be generic and occasionally awkward.
- **Long‑context fidelity**: full‑PDF map‑reduce still loses some detail.
- **Citations/attribution**: draft is narrative; citations are not yet auto‑formatted.

## 🔭 Roadmap / future improvements

- Switch to a stronger local summarizer or a **small LLM** for draft composition.
- Structured section templates (Intro/Related Work/Method/Experiments).
- Automatic **citations & references** (BibTeX export).
- Better PDF parsing (math/figures/tables) and deduplication.
- Optional **RAG with vector DB** for paragraph‑level grounding.

---

## 🧪 Reproduce a quick test

1. Paste a short idea (e.g., “BERT sentiment analysis IMDb”).  
2. Retrieve & rank papers.  
3. Keep **Top K = 5** and **disable full‑PDF** for speed.  
4. Generate and inspect the draft; then re‑run with full PDFs.

---

## 🔧 Troubleshooting

- **Slow PDF step** → uncheck *Use full PDFs*, increase later.  
- **CUDA errors** → verify driver + torch CUDA build; fall back to CPU by `set IDEA2PAPER_DEVICE=cpu`.  
- **Empty arXiv results** → try broader keywords; the app shows editable keywords.  
- **Model download blocked** → set HF mirror / allowlist domain in your network.

---

## 🙌 Credit requested

Please keep a link back to this repository if you reuse or fork the project, and cite in your course/project notes. No license is provided at this time; contributions welcome via PRs.

---

## ⭐ Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/transformers/)  
- [Sentence-Transformers / SPECTER](https://www.sbert.net/)  
- [arXiv API](https://arxiv.org/help/api/)  
- [PEGASUS-arXiv model](https://huggingface.co/google/pegasus-arxiv)
