# YuE2-3B Prompt Formatting Guide (Fact-Checked)

**Model:** `m-a-p/YuE2-3B` (Multimodal Art Projection) · **Task:** lyrics + style → full song (48 kHz stereo)
**Compiled:** 13 Oct 2026 · **License of weights:** CC BY-NC 4.0 (non-commercial)

---

## How to read the confidence labels

Not every tip you will find online comes from the model authors. Each rule below is tagged:

| Tag | Meaning |
|---|---|
| ✅ **Official** | Stated on the YuE2-3B model card or the official YuE2 repo |
| 🟦 **Documented integration** | Stated in ComfyUI's official docs (applies to that integration; likely but not guaranteed to hold elsewhere) |
| 🟨 **Inherited from YuE v1** | Stated in the official *first-generation* YuE README. Probably still applies to YuE2, but **I could not find it restated on the YuE2 card** |
| 🟧 **Community** | From third-party guides or LoRA authors. Useful, but not from the model authors |

> **Verification note.** The official `m-a-p/YuE2-3B` page could not be fetched directly in my environment. I read a mirror of the model card (`mrfakename/YuE2-3B`) whose text matched the snippets returned for the official page. The official GitHub README I could retrieve was the **YuE v1** version; its YuE2 "generation guide" pages were not retrievable. Treat 🟨 items as informed defaults rather than guarantees.

---

## 1. The two inputs

Every generation takes two text fields:

| Field | Purpose |
|---|---|
| `style` | Short description of the sound: genre, instruments, mood, voice |
| `lyrics` | Words to be sung, split into labeled sections |

✅ **Official.** The model card's example loads a JSON containing exactly `style` and `lyrics` (plus `seed`) and calls the pipeline with them. There is no separate "caption" field.

```python
from yue2 import YuE2Pipeline

pipe = YuE2Pipeline.from_pretrained("m-a-p/YuE2-3B", device="cuda")

song = pipe(style=style, lyrics=lyrics, cot="full", seed=42)
song.save("song.flac")
song.save_artifacts("outputs/song")  # ABC, tokens, latents, audio, settings
```

Install (per the model card): Linux, Python 3.10+, 24 GB NVIDIA GPU with BF16 support; the `yue2_infer` wheel is downloaded from the model repo.

---

## 2. Writing the `style` prompt

### Recommended contents

🟦 **ComfyUI docs** list these useful elements: genre, vocals, language, tempo, instruments (and the cover workflow adds mood). 🟨 The YuE v1 README says a stable tag prompt covers **genre, instrument, mood, gender, timbre**. The two lists overlap heavily, so cover as many as you can.

| Element | Examples |
|---|---|
| Genre | `pop`, `jazz-funk`, `cyber metal`, `folk` |
| Vocals (gender + tone) | `warm female vocal`, `expressive male vocal`, `airy vocal` |
| Language | `English`, `Chinese`, `Japanese` |
| Tempo | `96 BPM`, `slow`, `upbeat` |
| Instruments | `piano`, `electric bass`, `brushed drums`, `synths` |
| Mood | `uplifting`, `melancholic`, `intimate late-night atmosphere` |

### Format

Both of these are shown in official/documented sources, so both work:

```text
# ✅ Official model-card example (comma-separated phrases)
Jazz-funk, warm lead vocal, Rhodes piano, electric bass, tight drums

# 🟦 ComfyUI docs example (includes language + BPM)
English, warm female vocal, contemporary pop, 96 BPM, piano, rounded electric bass, restrained drums, clear diction

# 🟨 YuE v1 example (space-delimited tags)
inspiring female uplifting pop airy vocal electronic bright vocal vocal
```

### Additional guidance

- 🟦 **Keep musical instructions in `style`, not in the lyrics.**
- 🟨 Tag order is flexible (v1 README).
- 🟨 v1 shipped a `top_200_tags.json` and recommended choosing from it for stability. I did **not** find confirmation that YuE2 uses the same list, so treat it as a vocabulary hint, not a requirement.
- 🟨 v1 added `Mandarin` and `Cantonese` tags to disambiguate the two. Reasonable to try for YuE2; unconfirmed.
- 🟧 Community guide: put the most important words first; match the language of the lyrics to the genre/language in the style (e.g. don't pair "Korean pop" with English lyrics).

---

## 3. Writing the `lyrics` field

### Structure labels

Put a label in square brackets on its own line before each section.

| Label | Status |
|---|---|
| `[Verse]` `[Chorus]` `[Bridge]` `[Outro]` | 🟦 Shown in ComfyUI's YuE2 docs |
| `[Pre-Chorus]` | 🟧 Used in the community guide; also in the list of tags a community LoRA was trained on |
| `[Intro]` | 🟨 v1 found it **less stable**; recommends starting with `[Verse]` or `[Chorus]` |
| `[Interlude]`, `[Guitar Solo]` | 🟧 Community guide only; unverified |

```text
[Verse]
Morning light across the window
City waking down below

[Chorus]
Run with me into the sunlight
Leave the shadows far behind
```

### Formatting rules

| Rule | Status |
|---|---|
| Lyrics field contains **only** section tags and words meant to be sung | 🟦 ComfyUI docs; 🟧 community guide agrees |
| Separate sections with **one blank line** (`\n\n`) | 🟨 v1 README specifies two newline characters between sections; 🟧 community guide repeats it for YuE2 |
| Start with `[Verse]` or `[Chorus]` rather than a lyric-filled `[Intro]` | 🟨 v1 |
| Keep each section short; a section is roughly **30 seconds** of audio | 🟨 v1 README ("don't put too many words in a single segment"). Whether YuE2 uses the same ~30 s unit is unconfirmed |
| **Write repeated sections out in full**; don't use `(repeat chorus)` or `[Chorus x2]` | 🟧 Community guide. Sensible, since the model sings the text it's given, but not stated by the authors |
| Keep syllable counts similar on repeated lines | 🟧 Community guide |
| No production notes like `(drums build)` or `[key change]` in lyrics | 🟦 (only sung words) + 🟧 (explicit examples) |

### Language

🟨 v1 supported English, Mandarin, Cantonese, Japanese and Korean. 🟦 ComfyUI's YuE2 docs list English, Chinese and Japanese. 🔎 A community thread on the official repo asks "Supported languages?"; I found no authoritative YuE2 list, so test your language before committing to a long project.

---

## 4. Planning mode: the `cot` parameter

✅ **Official** (model card):

| Value | Behaviour | Use for |
|---|---|---|
| `cot="full"` | Plans melody **and** chords as an ABC score first (default) | Original songs |
| `cot="melody"` | Melody-only plan | **Covers** (card recommends it) |
| `cot="off"` | Generates directly with no symbolic plan | Quick experiments |

Related official options:

- `cfg_scale=1.2` is suggested on the card as something to *experiment* with for stronger text guidance. Semantic CFG defaults to **1.0** for full/melody and **1.01** for off.
- Pass your own score with `abc=Path("edited.abc").read_text(encoding="utf-8")`.
- `pipe.plan(...)` returns the plan before audio is generated, so you can edit it first.

```python
# Edit-the-score workflow (from the model card)
plan = pipe.plan(style=style, lyrics=lyrics, cot="full", seed=42)
plan.save("original_plan")
# edit a copy as edited.abc, then:
song = pipe(style=style, lyrics=lyrics, cot="full", seed=42,
            abc=Path("edited.abc").read_text(encoding="utf-8"))
```

---

## 5. Covers

✅ **Official** workflow:

1. Transcribe the source with **SheetSage2** and save the melody ABC **without chord symbols** → `melody.abc`.
2. Get the lyrics, organised into sections that match the recording → `cover_lyrics.txt`.
3. Generate with the **target** style and `cot="melody"`.

```python
cover = pipe(
    style="Jazz-funk, warm lead vocal, Rhodes piano, electric bass, tight drums",
    lyrics=Path("cover_lyrics.txt").read_text(encoding="utf-8"),
    abc=Path("melody.abc").read_text(encoding="utf-8"),
    cot="melody", seed=831001,
)
```

🟦 ComfyUI docs add: covers work best when the new lyrics stay close to the original phrasing and syllable count, since the melody is the link to the original. Note also that the card says `cot="melody"` does **not** strip chord symbols for you; use `cot="full"` if you want to supply harmony.

---

## 6. Instrumental tracks

The model authors' YuE2 card does not document an instrumental recipe. What exists:

- 🟧 **Community guide:** keep every section tag, leave the lyric lines under them empty, and drop vocal words from the style (e.g. add `instrumental`). It stresses that this differs from leaving `lyrics` entirely empty, because the tags drive the arrangement plan, and suggests `cot="full"`.

  ```text
  # style
  Indie pop, bright acoustic guitar, soft drums, hopeful, instrumental

  # lyrics
  [Verse]

  [Chorus]

  [Outro]
  ```
- 🟧 **Third-party LoRA** (`YuE2-instrumental-cot-full-loras`) accepts bare, untimed, or timed tags like `[verse 0:15-0:45]`. These timed tags are **specific to that LoRA**, not the base model.

Treat the empty-section approach as an experiment to validate on your own outputs.

---

## 7. Length and budget

🟦 **ComfyUI integration only:** Max Duration is capped at 900 s, with a budget of 25 semantic tokens per second of requested duration. A very long style, lyrics, or ABC plan consumes context and **shortens** the song rather than erroring. If you see a "token budget reached" warning, raise Max Duration, or shorten style/lyrics/ABC if a budget-reduction warning also appeared. The native Python pipeline's limits weren't documented in sources I could retrieve.

✅ **Official hardware figures:** a 24 GB GPU is sufficient without quantization; the card's benchmark shows a ~3.6-minute song in ~71 s on an RTX 4090 (peak VRAM ≈ 11 GiB).

---

## 8. Copy-paste template

```python
style = (
    "English, warm female vocal, indie pop, 100 BPM, "
    "acoustic guitar, soft drums, hopeful"
)

lyrics = """[Verse]
Streetlights flicker on an empty road
Every step feels lighter than it's ever showed

[Chorus]
We are rising with the morning sun
Nothing's over till the story's done
Hold my hand and we will find the way
Turning shadows into brighter days

[Verse]
Second verse lines go here
Keep the line lengths similar to the first

[Chorus]
We are rising with the morning sun
Nothing's over till the story's done
Hold my hand and we will find the way
Turning shadows into brighter days

[Outro]
Brighter days, brighter days
"""

song = pipe(style=style, lyrics=lyrics, cot="full", seed=1234)
song.save("song.flac")
```

Structure notes: begins with `[Verse]`, blank line between sections, chorus written out in full both times, only sung words in the lyrics.

---

## 9. Pre-flight checklist

- [ ] `style` covers genre, vocals, instruments, mood (tempo/language if it matters)
- [ ] No musical instructions inside `lyrics`
- [ ] Every section has a `[Label]` on its own line
- [ ] Blank line between sections
- [ ] First section is `[Verse]` or `[Chorus]`
- [ ] Repeated sections written out in full
- [ ] Sections short (a handful of lines)
- [ ] Lyrics language consistent with the style prompt
- [ ] `cot` chosen deliberately (`full` for originals, `melody` for covers)
- [ ] Fixed `seed` set if you want reproducible comparisons

---

## 10. Open questions (not verifiable from sources I could reach)

1. Whether YuE2 uses the v1 `top_200_tags.json` vocabulary.
2. Whether the v1 "blank line between sections" and "~30 s per section" rules are formally required in YuE2.
3. The authoritative list of supported singing languages.
4. Whether acapella/vocal-only output is supported (an unanswered community question on the model repo).
5. Exact structure labels beyond `Verse / Chorus / Bridge / Outro` that the base model was trained on.

For these, check the official model card and the `multimodal-art-projection/YuE` repo's YuE2 generation guide directly, or test empirically with a fixed seed.

---

## Sources

| Source | Used for | Tier |
|---|---|---|
| YuE2-3B model card (read via mirror `huggingface.co/mrfakename/YuE2-3B`; official: `huggingface.co/m-a-p/YuE2-3B`) | Inputs, `cot`, `abc`, `cfg_scale`, covers, hardware, license | ✅ |
| ComfyUI YuE2 docs: `docs.comfy.org/tutorials/audio/yue2/yue2` | Style elements, lyric tags, budget behaviour | 🟦 |
| YuE (v1) README: `github.com/multimodal-art-projection/YuE` | Five style components, top-200 tags, `\n\n`, ~30 s, `[intro]` caution | 🟨 |
| Dawid Wasowski, "Generate Music with YuE2" (12 Sep 2026): `dav.one/how-to-generate-music-with-yue2/` | Repeat-in-full, syllables, language sync, instrumental recipe | 🟧 |
| `Mothersuperior/YuE2-instrumental-cot-full-loras` (Hugging Face) | LoRA-specific timed tags | 🟧 |
