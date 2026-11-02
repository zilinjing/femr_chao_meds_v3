"""Check the shared tokenizer and compare the task code sets of the six models.

Works for any dataset prepared with script/lib.sh (MIMIC, CUMC): pass its three paths.
Read-only. Usage (tte env):
    python check_tokenization.py --out_root <OUT_ROOT> --meta_dir <META_DIR> --meds_reader <MEDS_READER>
                                 [--num_threads 32] [--no_event_counts]

A. every model's tokenizer/dictionary.msgpack is byte-identical to META_DIR/tokenizer;
   vocabulary composition and numeric bins per column
B. task code sets (MOTOR, TPP, MTPP, NTP, SurvivEHR, ICareFM) and their differences; each
   differing code is shown with its event counts by kind (train+tuning split) and the filter
   message of each model's prepare log. Ancestor-only codes (ontology) have no raw events.
   --no_event_counts skips the event-count pass (slow on large datasets).
"""
import argparse
import collections
import filecmp
import functools
import glob
import pathlib
import pickle

import msgpack
import numpy as np
import pandas as pd

MODELS = ["motor", "tpp", "mtpp", "ntp", "survivehr", "icarefm"]
KINDS = ["none", "numeric", "text", "numeric+text"]


def event_kind(event):
    has_num = event.numeric_value is not None
    has_text = event.text_value is not None
    return KINDS[has_num + 2 * has_text]


def count_map(subjects):
    counts = collections.defaultdict(lambda: [0, 0, 0, 0])
    for subject in subjects:
        for event in subject.events:
            counts[event.code][KINDS.index(event_kind(event))] += 1
    return dict(counts)


def count_agg(a, b):
    for code, c in b.items():
        if code in a:
            a[code] = [x + y for x, y in zip(a[code], c)]
        else:
            a[code] = c
    return a


def event_counts(meds_reader_path, split_csv, num_threads):
    """Event counts per code and kind over the train+tuning subjects (the data the tasks are fit on)."""
    import meds_reader

    splits = pd.read_csv(split_csv)
    train_ids = splits[splits.split_name == "train"].subject_id.tolist()
    with meds_reader.SubjectDatabase(str(meds_reader_path), num_threads=num_threads) as db:
        db = db.filter(train_ids)
        return functools.reduce(count_agg, db.map(count_map), {})


def filter_messages(model_dir):
    """Code -> filter message printed by the task fitting (latest prepare log of the model)."""
    logs = sorted(glob.glob(str(model_dir / "logs" / "prepare_*.log")))
    # Codes can contain spaces, so cut on the fixed text around them instead of on whitespace.
    # MOTOR/TPP print "<prefix> <code> <frac> <censored> <events> <mean>"; MTPP prints "<prefix> <code>, frac_events=..."
    # or "<prefix> <code>: frac_events=...".
    rules = [
        ("Ran into task of rate 0? ", "rate==0", lambda rest: rest.rsplit(" ", 4)[0]),
        ("Ran into very rare task with less than 10 occurrences ", "frac_events<1/1000", lambda rest: rest.rsplit(" ", 4)[0]),
        ("Ran into task of rate 0: ", "rate==0", lambda rest: rest.split(", frac_events=")[0]),
        ("Filtered out numerical task ", "numeric: frac<1/1000 or value samples<threshold", lambda rest: rest.split(": frac_events=")[0]),
        ("Filtered out rare non-numerical task ", "frac_events<1/1000", lambda rest: rest.split(": frac_events=")[0]),
    ]
    messages = {}
    for log in logs:
        with open(log) as f:
            for line in f:
                line = line.rstrip("\n")
                for prefix, reason, get_code in rules:
                    if line.startswith(prefix):
                        rest = line[len(prefix):]
                        messages[get_code(rest)] = f"{reason} | {rest[-110:]}"
    return messages


def show(name, codes, counts, messages, original, limit=20):
    codes = sorted(codes)
    print(f"  {name}: {len(codes)}")
    if counts is not None and codes:
        kind_totals = collections.Counter()
        for code in codes:
            c = counts.get(code, [0, 0, 0, 0])
            kind_totals[tuple(k for k, n in zip(KINDS, c) if n > 0)] += 1
        summary = ", ".join(f"{'+'.join(k) or 'no events'}-kinds: {n}" for k, n in kind_totals.most_common())
        print(f"    event kinds present (codes): {summary}")
    for code in codes[:limit]:
        c = counts.get(code) if counts is not None else None
        c = dict(zip(KINDS, c)) if c else None
        tag = "" if code in original else "[ancestor-only] "
        msg = " || ".join(f"{model}: {m[code]}" for model, m in messages.items() if code in m)
        print(f"    {code:45s} {tag}{c if c is not None else ''} {msg}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_root", default="/shared/share_mala/zilin/storage/ora/models")
    parser.add_argument("--meta_dir", default="/shared/share_mala/zilin/storage/ora/metadata")
    parser.add_argument("--meds_reader", default="/shared/share_mala/zilin/storage/ora/meds_v0.6_reader")
    parser.add_argument("--num_threads", type=int, default=32)
    parser.add_argument("--no_event_counts", action="store_true")
    args = parser.parse_args()
    out_root, meta_dir = pathlib.Path(args.out_root), pathlib.Path(args.meta_dir)

    # ---- A. shared tokenizer ----
    print("== A. shared tokenizer")
    shared = meta_dir / "tokenizer" / "dictionary.msgpack"
    for model in MODELS:
        local = out_root / model / "tokenizer" / "dictionary.msgpack"
        same = local.exists() and filecmp.cmp(shared, local, shallow=False)
        print(f"  {model:10s} identical to shared: {same}")
    with open(shared, "rb") as f:
        dictionary = msgpack.load(f)
    vocab = dictionary["vocab"]
    print(f"  vocab entries: {len(vocab)}  by type: {dict(collections.Counter(v['type'] for v in vocab))}")
    code_tokens = [v["code_string"] for v in vocab if v["type"] == "code"]
    original_codes = list(dictionary["original_codes"])
    print(f"  code tokens: {len(code_tokens)}  original_codes: {len(original_codes)}  "
          f"original_codes == code tokens (as sets): {set(original_codes) == set(code_tokens)}  "
          f"original_codes ⊆ code tokens: {set(original_codes) <= set(code_tokens)}")
    print(f"  duplicates: code tokens {len(code_tokens) - len(set(code_tokens))}, "
          f"original_codes {len(original_codes) - len(set(original_codes))}")
    print("  (without an ontology, original_codes == code tokens; with one, ancestor-only codes are input tokens only)")

    # Numeric bins per column. Requested bins = int(1000 x the column's share of numeric values);
    # duplicate quantile edges are merged, so a column can get fewer.
    print("\n  numeric bins per column (interior edges: min / 25% / median / 75% / max):")
    numeric = collections.defaultdict(list)
    for v in vocab:
        if v["type"] == "numeric":
            numeric[v["property"]].append(v["val_start"])
    for prop, starts in sorted(numeric.items(), key=lambda kv: -len(kv[1])):
        edges = sorted(starts)[1:]  # the first bin starts at -inf
        q = np.percentile(edges, [0, 25, 50, 75, 100]) if edges else []
        print(f"    {prop:22s} {len(starts):5d} bins   " + " / ".join(f"{x:.4g}" for x in q))
    text = collections.Counter(v["property"] for v in vocab if v["type"] == "text")
    print(f"  text entries per column: {dict(text.most_common())}")

    # ---- B. task code sets ----
    print("\n== B. task code sets")
    tasks = {}
    for model in MODELS:
        with open(out_root / model / f"{model}_task.pkl", "rb") as f:
            tasks[model] = pickle.load(f)
    motor = set(tasks["motor"].pretraining_task_codes)
    tpp = set(t[0] for t in tasks["tpp"].pretraining_task_info)
    mtpp = set(tasks["mtpp"].pretraining_task_codes)
    mtpp_num = set(tasks["mtpp"].numerical_task_to_index_map)
    mtpp_non = set(tasks["mtpp"].task_to_index_map)
    ntp_size = tasks["ntp"].clmbr_vocab_size
    surv_event = set(tasks["survivehr"].original_codes)
    surv_value = set(tasks["survivehr"].numerical_task_codes)
    icare = set(tasks["icarefm"].target_codes)

    print(f"  MOTOR {len(motor)} | TPP {len(tpp)} | MTPP {len(mtpp)} (numeric {len(mtpp_num)}, "
          f"non-numeric {len(mtpp_non)}) | NTP {ntp_size} | SurvivEHR event {len(surv_event)}, "
          f"value {len(surv_value)} | ICareFM {len(icare)}")
    print(f"  MOTOR time bins: {len(tasks['motor'].time_bins)} edges; "
          f"TPP time_bins shape {tasks['tpp'].time_bins.shape}; MTPP value_bins shape {tasks['mtpp'].value_bins.shape}")

    counts = None
    if not args.no_event_counts:
        print("  counting events per code and kind (train+tuning split) ...")
        counts = event_counts(args.meds_reader, out_root / "motor" / "main_split.csv", args.num_threads)
    messages = {m: filter_messages(out_root / m) for m in ("motor", "tpp", "mtpp")}
    original = set(original_codes)
    show_ = functools.partial(show, counts=counts, messages=messages, original=original)

    print("\n  H1: MOTOR == TPP")
    print(f"    equal: {motor == tpp}")
    show_("MOTOR - TPP", motor - tpp)
    show_("TPP - MOTOR", tpp - motor)

    print("\n  H2: MTPP vs TPP")
    print(f"    |MTPP ∩ TPP| = {len(mtpp & tpp)}  (MTPP non-numeric ⊆ TPP: {mtpp_non <= tpp})")
    show_("MTPP - TPP", mtpp - tpp)
    print(f"    of which MTPP numeric: {len((mtpp - tpp) & mtpp_num)}")
    show_("TPP - MTPP", tpp - mtpp)
    print(f"    MTPP numeric ∩ TPP (numeric codes that also have value-less events): {len(mtpp_num & tpp)}")

    print("\n  H3: NTP == SurvivEHR event codes")
    print(f"    NTP size {ntp_size} == len(original_codes) {len(original_codes)}: {ntp_size == len(original_codes)}")
    print(f"    SurvivEHR event codes == original_codes (same order): {list(tasks['survivehr'].original_codes) == original_codes}")
    show_("code tokens - original_codes (input-only codes)", set(code_tokens) - original)

    print("\n  H4: ICareFM vs MTPP numeric")
    print(f"    ICareFM == SurvivEHR value codes: {icare == surv_value}")
    print(f"    ICareFM ⊇ MTPP numeric: {icare >= mtpp_num}   |∩| = {len(icare & mtpp_num)}")
    show_("MTPP numeric - ICareFM", mtpp_num - icare)
    show_("ICareFM - MTPP numeric", icare - mtpp_num)
    print(f"    ICareFM - MTPP numeric that are MTPP non-numeric tasks: {len((icare - mtpp_num) & mtpp_non)}")

    print("\n  Targets vs all code tokens")
    show_("code tokens in no MOTOR/TPP/MTPP task", set(code_tokens) - motor - mtpp, limit=10)


if __name__ == "__main__":
    main()


'''
python $S/check_tokenization.py \
  --out_root /shared/share_mala/zilin/storage/ora/models \
  --meta_dir /shared/share_mala/zilin/storage/ora/metadata \
  --meds_reader /shared/share_mala/zilin/storage/ora/meds_v0.6_reader | tee check_v2.txt
'''