# Agent Spec: One Base Template Prompt (YuE2) from the 177 Captions

**For:** an AI agent working in the repo folder described in section 2. **This is a spec, not an implementation.** Numbers marked *(default)* apply unless the user overrides them before the run. Never change a threshold after seeing results; if one must change, log why.

## 1. Goal and working assumption
Reduce the captions to **one base template prompt** in the YuE2 `style` field format (see Step 6): the concepts that survive across most data points. The user then appends or removes tags.

Assumption (stated, not tested): the captioner did an acceptable job but also made things up and added noise. Idiosyncratic details will not repeat across most data points, so **consensus is the filter**. The Suno prompt, audio, and captioner are not used and must not influence which tags are kept.

## 2. Inputs

| Input | Status | If missing |
|---|---|---|
| `arabmaqamrock.csv` with a `caption` column (177 rows) | **Required** | Cannot start. Check the header on load; ignore any unnamed index column. |
| `yue2-3b-prompt-guide.md`: the YuE2-3B prompt formatting guide | **Required reading before Step 6** (already in the repo) | Cannot fix the output format; stop and ask |
| `yue2_tags.json` (or `.txt`/`.csv`): an optional vocabulary hint, e.g. the YuE v1 top-200 tag list. Not confirmed to apply to YuE2 | Optional | Skip tag matching; use plain descriptive phrases (Step 6) |
| Human labels (checkpoints C1, C2 in section 4) | Optional | Agent labels itself; every pass/fail is marked **self-labelled, unverified** |

A caption ID column is not needed: use the row index. Sanity check on load: no nulls, report exact-duplicate captions, word-count range (previous run: 246-554 words, mean ~392). Use `.copy()` after selecting columns.

### Repo layout

```
yue2_prompt_template/
├─ arabmaqamrock.csv            input (required)
├─ yue2_style_tag_plan.md       this spec
├─ yue2-3b-prompt-guide.md      input (required reading for Step 6)
├─ yue2_tags.json               input (optional vocabulary hint; any of .json/.txt/.csv)
├─ yue2_base_template.ipynb     agent creates
├─ requirements.txt             agent creates
├─ labels/
│  ├─ c1_boundary_sheet.csv     agent writes (blind sentences, 25 captions)
│  ├─ c1_boundary_labels.csv    user returns: caption_id, sentence numbers where a new passage starts
│  ├─ c2_pair_sheet.csv         agent writes (200 pairs + 20 opposite probes, method hidden)
│  └─ c2_pair_labels.csv        user returns: pair_id, same|different
├─ outputs/
│  ├─ base_template.txt
│  └─ template_table.csv
└─ archive/                     optional; move maqam_prompt_generator.py here
```

`maqam_prompt_generator.py` and the earlier notebook are **not inputs**. Do not read the script, and do not use anything in it to decide which tags to keep (rule 3, section 6). Caption ID is the CSV row index. If the CSV has no other columns, that is expected.

## 3. Environment check (first action)
Confirm availability of: spaCy with an English model, scikit-learn, pandas, numpy, and one sentence-embedding model.
- **Embedding models: two candidates, both local, both run in Step 3 as separate methods.** Used only for merging in Step 3 (and for tag matching in Step 6 if the user supplies a vocabulary hint, using whichever model won Step 3). Used for nothing else.
  1. `BAAI/bge-small-en-v1.5` via `sentence-transformers` (33M parameters, 384-d, English, 512-token limit). The established baseline.
  2. `google/embeddinggemma-2` via `sentence-transformers`, **text-only**: load with `config_kwargs={"vision_config": None, "audio_config": None}` so the vision and audio encoders (together ~470M parameters) are not loaded. Rules from its model card: run in `bfloat16` or `float32`, **never `float16`** (it returns NaN or silently degraded embeddings); apply the symmetric `Clustering` (or `SentenceSimilarity`) prompt name to every phrase; L2-normalise; if truncating dimensions, re-normalise afterwards. Install with `pip install -U sentence-transformers transformers`.
  - GGUF builds (e.g. `unsloth/embeddinggemma-2-GGUF`, Q8_0 ≈ 310 MB) are a fallback **only** if PyTorch is unavailable. They need llama.cpp with mean pooling and a runtime that preserves the model's embedding projection, the task prefix must be added by hand, and their quality loss from quantisation is unmeasured here. Prefer the `sentence-transformers` route.
  - Do not substitute a larger model or call any hosted API or generative LLM; the user's text must stay local. Record exact model names and revisions in the notebook.
- If something can't be installed, use these fallbacks and record them: NLTK or rule-based lemmatising for spaCy, character n-gram TF-IDF for the embedding model.

## 4. Steps

### Step 1. Passage segmentation (the unit of counting)
**Why:** captions appear to be several concatenated passages, each with its own summary sentence naming a genre. Per-caption counting would give every genre label in a caption 100%, so consensus would be meaningless.
**Procedure:**
1. Split into sentences. Build a boundary detector by inspecting captions and printing candidate sentences. Starting hypotheses (unvalidated, from earlier loose probes; confirm or discard): a new passage starts at a sentence opening with "An/The ..." that names the piece ("track", "anthem", "ballad", "arrangement", etc.) near its start; passages are around 4 sentences long; passage count per caption is about 3-6.
2. Draw a stratified sample of 25 captions (by detected passage count, including captions where the detector found nothing) **after** the detector exists. Export blind sheets: each caption as numbered sentences, with no detector output shown.
3. **C1 (human):** for each of the 25, list the sentence numbers where a new passage begins. Estimated 20-25 minutes.
4. Score: a boundary counts as correct only if the sentence index matches exactly. **Pass: boundary F1 ≥ 0.95 *(default)*.**
5. On failure: iterate on unlabelled captions only, then draw a fresh sample of 15 not used before. If still failing, fall back to fixed 4-sentence windows and label every downstream result "windowed".

**Opening passage rule:** compute each passage's concept set (after Step 3). If the first passage's top-20 concepts overlap with the top-20 of later passages by Jaccard < 0.5, vote on it separately, exclude it from the base template, and list its concepts in a separate "opening only" table. Otherwise pool it.

### Step 2. Concept extraction and buckets
- Per passage, extract (modifier, head-noun) phrases with the dependency parse (amod/compound + noun). Lowercase, lemmatise, drop determiners and adverbs, keep up to 2 modifiers.
- **Bucket assignment:** list the 150 most frequent head nouns; assign each to genre / instrument / vocal / mood / production / other using the head noun (modifiers of piece-naming heads like "track" or "anthem" go to genre). Output this head-noun table for the user to veto. Heads left in "other" never enter the template.
- Never merge or cluster across buckets.

### Step 3. Merge wording variants
**Why:** earlier probes found one instrument in most passages but in about 150 distinct wordings, so no single wording reached a high share.
**Candidate methods**, all evaluated on the same labelled pairs:
1. Exact phrase match (baseline, expected to fragment).
2. Lemma + head-modifier matching ("heavily distorted electric guitars" and "distorted guitars" share head and modifier).
3. Embedding similarity within a bucket, **run once per candidate embedding model** (section 3), so each model counts as its own method (3a, 3b): embed phrases, agglomerative clustering with a conservative cosine distance threshold. Start at 0.15 for each model. Distance scales differ between models, so if inspecting 20 sample pairs shows 0.15 merges nothing or nearly everything for a model, set that model's threshold from that inspection **before** the C2 pairs are sampled, and never loosen it afterwards to pass. Clusters are merge suggestions.
4. Hybrid: method 2 first, then the best embedding method on what remains.

**Opposite-pair guard (all methods):** never merge across male/female, clean/distorted, fast/slow, soft/loud, acoustic/electric, major/minor, and any further opposite pair the agent finds in the head-noun table. Embeddings place opposites close together, so this guard is mandatory.

**Pair sampling for C2:** 200 pairs drawn from pairs each method proposes to merge (stratified by method), plus 20 agent-seeded opposite-pair probes.
**C2 (human):** mark each pair same or different. Estimated 15 minutes.
**Pass: merge precision ≥ 95% (report the Wilson interval), zero opposite pairs merged *(default)*.** Choose the passing method with the best recall; if none passes, use the strictest passing method and report the lower recall.
**Naming:** each merged concept is named by its most frequent surface form, not the cluster medoid. Save the member list of every cluster in the output table, and spot-check the 10 largest.

### Step 4. Count and select
- Each passage votes once per concept, however often it repeats.
- Report per concept: share of passages, share of captions containing it at least once.
- **Cut-off *(default)*: ≥ 60% of passages and ≥ 60% of captions.** Plot the share distribution and report any natural gap, but do not move the cut-off unless the user says so.
- **Stability:** run 100 random half-splits of the captions. A concept is *stable* if kept in ≥ 90% of splits. Kept-but-unstable concepts go to the near-miss list, not the base. Also report which concepts change status at ±10 points.
- **Generic flag:** any concept in ≥ 90% of passages with no instrument, vocal, or genre head noun (a standalone adjective like an intensifier) is flagged generic. Do not drop it; the user decides.

### Step 5. Mutually exclusive concepts
Two concepts in the same bucket are exclusive if their observed passage co-occurrence is below 10% of the count expected under independence, where the expected count is at least 5. For each exclusive set, put the most common concept in the template and list the others as alternatives.

### Step 6. Format as a YuE2 `style` prompt and abridge
**Read `yue2-3b-prompt-guide.md` first.** It labels each rule's confidence (official / documented integration / inherited from YuE v1 / community). Follow only what it states, keep those labels in the output notes, and do not present inherited or community rules as official. Facts used here:
- YuE2-3B takes a `style` string and a `lyrics` string; there is no separate caption or negative-prompt field. (Official, per the guide. A mirror of the model card was read by the guide's author, who could not fetch the official page; a separate web check I made confirmed `style`, `lyrics`, and comma-separated examples via a community port but not the official card itself.)
- **Format:** comma-separated descriptive phrases, e.g. `Jazz-funk, warm lead vocal, Rhodes piano, electric bass, tight drums` (official model-card example per the guide). Space-delimited v1 tags also appear in the guide but are inherited from YuE v1; do not use them.
- **Elements to cover when the data supports them:** genre, vocals (gender + tone), language, tempo, instruments, mood (documented integration list; YuE v1's five components, genre/instrument/mood/gender/timbre, overlap and are only a hint).
- No confirmed YuE2 tag list exists in the guide. The v1 `top_200_tags.json` is at most a vocabulary hint.

**Procedure:**
1. Give each kept concept a slot: genre, vocal, instrument, mood, production, language, or tempo.
2. Write each as a short natural phrase in the guide's style (e.g. `warm male vocal`, not a bare adjective). If the user supplied a vocabulary hint, prefer a listed phrase when one is a clean match (by embedding similarity then judgment) and mark the rest **out-of-hint**; if there is no hint file, skip matching and mark every phrase **free-form**. If a concept has no sensible phrase (e.g. "vibrato" as a standalone tag), list it as **unmapped**.
3. **Language:** if a language concept is in the base (the captions mention Arabic lyrics), include it as the language phrase but flag **language support unverified**. The guide lists English, Chinese and Japanese (documented integration) and says there is no authoritative language list; no source I read lists Arabic. The user must test Arabic output before relying on the template.
4. **Tempo:** the captions give no BPM. Do **not** invent one. Include a tempo phrase only if a qualitative tempo concept passes the cut-off. Otherwise leave the slot empty and say so; the user may append a BPM themselves.
5. Write all phrases positively. Do not add negations or exclusions ("no synth pads"); the documented inputs have nowhere for them.
6. **Budget *(default)*: 8-14 phrases.** Keep the style short: the guide notes that very long style text can consume context in one integration. Rank by passage share but ensure each slot has a phrase if any concept qualified. When over budget, cut the lowest-share phrases first and list them under near misses.
7. **Order *(default, unverified)*:** genre first (as in the official example), then the remaining phrases by passage share, highest first. The guide says order is flexible (inherited from v1) and that putting important words first is a community tip. Do not claim either as proven.

### Step 7. Outputs
1. `base_template.txt`: the one-line comma-separated `style` string.
2. `template_table.csv`: concept, bucket, slot, yue2_phrase, mapping (`hint-matched` / `out-of-hint` / `free-form` / `unmapped`), passage_share, caption_share, stable, generic, alternatives, cluster members, notes (include confidence labels from the guide).
3. Near-miss list (just under the cut-off or unstable) in the same CSV with a `layer` column (`base`, `near_miss`, `opening_only`).
4. `yue2_base_template.ipynb` that runs the whole pipeline end to end, plus `requirements.txt`. No other documents.

## 5. Human checkpoints (total ≈ 40-45 minutes)

| ID | Task | Time | If skipped |
|---|---|---|---|
| C1 | Mark passage boundaries on 25 blind captions | ~20-25 min | Agent labels; Step 1 marked unverified |
| C2 | Judge 200 phrase pairs same/different | ~15 min | Agent labels; Step 3 marked unverified |
| C3 | Veto glance at the head-noun bucket table and the final table | ~5 min | Agent proceeds, flags assignments it was unsure about |

## 6. Rules for the agent
1. Never derive a rule or lexicon entry from a single caption; cite counts across all 177.
2. State each test, its pass criterion, and what happens on failure before running. Report failures.
3. Do not use the prompt, audio, or captioner to include or exclude a tag.
4. Mark every verdict as **verified** (human-labelled) or **self-labelled, unverified**.
5. If a stage's requirement can't be met, use the stated fallback and say so in the output.

## 7. Done means
`base_template.txt` and `template_table.csv` exist; every pass/fail has a recorded value; every tag has a passage share, caption share, and stability flag; unmapped, free-form/out-of-hint, and generic items are listed; the Arabic-language and empty-tempo notes are present; the notebook reruns end to end.

## 8. Not verified by this plan
- Whether YuE2 responds to each phrase, and whether the captions match the audio. The user tests that by generating; it falls outside the stated assumption.
- **Whether YuE2 can sing Arabic.** No source in the guide lists it. Test before relying on any language phrase.
- Whether the YuE v1 tag list, tag order, or any 🟨/🟧 rule in the guide applies to YuE2. The guide marks these as unconfirmed.
- The guide itself is a compiled secondary source whose author could not fetch the official model card directly; treat its labels as its own confidence ratings, not guarantees.
