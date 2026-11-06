"""LLM-as-judge for story point estimates on the 100 recent-Moodle annotation items.

The judge (Claude, called through the Claude Code CLI in headless mode: `claude -p`) sees exactly the
text the human estimators saw (items_shown.csv) and the same 12 reference examples with Moodle's
recorded story points. Three tasks:

  pointwise  rate one candidate estimate 1-5 (1 = clearly unreasonable, 5 = clearly right).
             Candidates: the recorded story point, GPT2SP (seed 1, rounded to a whole number, min 1),
             and the Median baseline (2). Two prompt wordings (v1, v2) to measure prompt sensitivity.
  pairwise   GPT2SP vs Median baseline, shown as A/B in BOTH orders (position bias); items where the
             two are equal after rounding are skipped.
  estimate   the judge's own estimate (LLM as an estimator, for comparison with GPT2SP and people).

Every call is repeated (--reps) to measure run-to-run variation. Calls are cached in
results/llm_judge/raw.jsonl, keyed by (task, variant, item, candidate/order, rep), so the script can
be stopped and resumed. No tools are enabled and the default system prompt is replaced.

Usage: python llm_judge/judge.py [--model sonnet] [--reps 3] [--workers 8]
"""
import argparse
import json
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ANN = ROOT / "new_data" / "annotation"
OUT = ROOT / "results" / "llm_judge"
GPT2SP_PREDS = ROOT / "results" / "gpt2sp" / "moodle_recent_within" / "moodle_recent__moodle_recent__seed1_preds.csv"
BASE_PREDS = ROOT / "results" / "baselines" / "moodle_recent_within" / "preds" / "moodle_recent__moodle_recent_preds.csv"
ANCHORS = ["MDL-73091", "MDL-78833", "MDL-77103", "MDL-81598", "MDL-81122", "MDL-65205",
           "MDL-80744", "MDL-67287", "MDL-80960", "MDL-77558", "MDL-81031", "MDL-80894"]
MAX_DESC = 1500

SYSTEM = ("You are an experienced member of an agile software team. You review story point estimates for "
          "issues in the Moodle open-source learning platform. Story points measure relative effort on "
          "this team's own scale, so compare each issue with the team's reference examples. "
          "Reply with a single JSON object and nothing else.")

TASKS = {
    ("pointwise", "v1"): ("Proposed estimate: {x} story points.\n\nIs this estimate reasonable for this issue on "
                          "the Moodle team's scale? Rate it from 1 to 5: 1 = clearly unreasonable, 2 = probably "
                          "wrong, 3 = plausible, 4 = probably right, 5 = clearly right.\n"
                          'Reply as JSON: {{"score": <1-5>, "reason": "<one sentence>"}}'),
    ("pointwise", "v2"): ("In planning poker a teammate proposes {x} story points for this issue.\n\nHow strongly "
                          "would you agree with this number, given the reference examples? 1 = I would strongly "
                          "push back, 3 = I could accept it, 5 = I fully agree.\n"
                          'Reply as JSON: {{"score": <1-5>, "reason": "<one sentence>"}}'),
    ("pairwise", "v1"): ("Two estimates were proposed for this issue.\nEstimate A: {a} story points.\n"
                         "Estimate B: {b} story points.\n\nWhich estimate is better supported by the issue text, "
                         "on the Moodle team's scale?\n"
                         'Reply as JSON: {{"choice": "A" or "B", "reason": "<one sentence>"}}'),
    ("estimate", "v1"): ("Estimate this issue in story points on the Moodle team's scale (a whole number; the team "
                         "mostly uses 1, 2, 3, 5, 6 and 10).\n"
                         'Reply as JSON: {{"estimate": <whole number>, "reason": "<one sentence>"}}'),
}


def shorten(text):
    text = "" if pd.isna(text) else str(text).strip()
    return text if len(text) <= MAX_DESC else text[:MAX_DESC].rstrip() + " [...text cut here]"


def reference_block():
    m = pd.read_csv(ROOT / "new_data" / "moodle_recent" / "marked_data" / "moodle_recent.csv").set_index("issuekey")
    a = m.loc[ANCHORS].sort_values("storypoint", kind="stable")
    parts = [f"[{int(r.storypoint)} story points] {r.title}\n{shorten(r.description)}" for r in a.itertuples()]
    return "Reference examples (older Moodle issues with the team's recorded story points):\n\n" + "\n\n---\n\n".join(parts)


def load_items():
    items = pd.read_csv(ANN / "items_shown.csv")
    key = pd.read_csv(ANN / "answer_key.csv", usecols=["item_id", "storypoint"])
    g = pd.read_csv(GPT2SP_PREDS).rename(columns={"predicted": "gpt2sp_raw"})[["issuekey", "gpt2sp_raw"]]
    b = pd.read_csv(BASE_PREDS)[["issuekey", "median_pred"]]
    d = items.merge(key, on="item_id").merge(g, on="issuekey").merge(b, on="issuekey")
    d["gpt2sp"] = d.gpt2sp_raw.round().clip(lower=1).astype(int)
    d["median"] = d.median_pred.round().astype(int)
    d["recorded"] = d.storypoint.astype(int)
    return d


def issue_block(r):
    return f"Issue to estimate\nType: {r.issuetype}\nTitle: {r.title}\nDescription:\n{r.description_shown if isinstance(r.description_shown, str) else '(empty)'}"


def build_jobs(d, reps):
    jobs = []
    for rep in range(1, reps + 1):  # rep-major order: one full pass first, so partial runs are balanced
        for r in d.itertuples():
            jobs.append(dict(task="estimate", variant="v1", item_id=r.item_id, cand="-", rep=rep,
                             body=TASKS[("estimate", "v1")]))
            if r.gpt2sp != r.median:
                for order, (a, b) in (("gpt2sp_first", (r.gpt2sp, r.median)), ("median_first", (r.median, r.gpt2sp))):
                    jobs.append(dict(task="pairwise", variant="v1", item_id=r.item_id, cand=order, rep=rep,
                                     body=TASKS[("pairwise", "v1")].format(a=a, b=b)))
            for variant in ("v1", "v2") if rep == 1 else ("v1",):
                for cand in ("recorded", "gpt2sp", "median"):
                    jobs.append(dict(task="pointwise", variant=variant, item_id=r.item_id, cand=cand, rep=rep,
                                     body=TASKS[("pointwise", variant)].format(x=getattr(r, cand))))
    return jobs


def key_of(j):
    return f"{j['task']}|{j['variant']}|{j['item_id']}|{j['cand']}|{j['rep']}"


def call_claude(model, prompt):
    cmd = ["claude", "-p", "--model", model, "--output-format", "json", "--tools", "",
           "--system-prompt", SYSTEM, "--no-session-persistence", "--strict-mcp-config",
           "--disable-slash-commands"]
    for attempt in range(4):
        p = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=300)
        try:
            out = json.loads(p.stdout)
            if not out.get("is_error"):
                return out
        except json.JSONDecodeError:
            pass
        time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"claude failed: {p.stdout[-300:]} {p.stderr[-300:]}")


def parse(text):
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        return json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sonnet")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="only the first N jobs (smoke test)")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    raw = OUT / "raw.jsonl"
    done = set()
    if raw.exists():
        done = {json.loads(l)["key"] for l in raw.open()}
    d = load_items()
    d.drop(columns=["description_shown"]).to_csv(OUT / "items_with_candidates.csv", index=False)
    ref = reference_block()
    rows = {r.item_id: r for r in d.itertuples()}
    jobs = [j for j in build_jobs(d, args.reps) if key_of(j) not in done]
    if args.limit:
        jobs = jobs[: args.limit]
    print(f"{len(done)} cached, {len(jobs)} to run with {args.workers} workers", flush=True)

    lock = threading.Lock()

    def run(j):
        prompt = f"{ref}\n\n=====\n\n{issue_block(rows[j['item_id']])}\n\n=====\n\n{j['body']}"
        out = call_claude(args.model, prompt)
        rec = {k: j[k] for k in ("task", "variant", "item_id", "cand", "rep")}
        rec.update(key=key_of(j), parsed=parse(out.get("result")), text=out.get("result"),
                   model=next(iter(out.get("modelUsage", {}) or {"?": 0})), cost_usd=out.get("total_cost_usd"))
        with lock:
            with raw.open("a") as f:
                f.write(json.dumps(rec) + "\n")
        return rec

    n = 0
    with ThreadPoolExecutor(args.workers) as ex:
        for fut in as_completed([ex.submit(run, j) for j in jobs]):
            n += 1
            try:
                fut.result()
            except Exception as e:  # keep going; failed jobs are retried on the next run
                print("FAILED:", e, flush=True)
            if n % 50 == 0:
                print(f"{n}/{len(jobs)}", time.strftime("%H:%M:%S"), flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
