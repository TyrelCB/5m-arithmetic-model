"""Three evaluators for the English model.

  (1) grammar   -- generate N sentences, score well-formedness with eng_grammar
  (2) nextword  -- given a held-out prefix, is the model's next WORD admissible?
  (3) minpairs  -- BLiMP-style: does the model assign higher likelihood to the
                   grammatical member of a minimal pair?

All three measure conformance to synth_word.py's grammar, not English.
"""
import argparse, random, re, sys, torch
import synth_word as sw
import eng_grammar as G

def load(arch, ckpt):
    src = open(arch).read().split("if __name__")[0]
    ns = {}; exec(compile(src, arch, "exec"), ns)
    m = ns["ModernGPT"]().cuda().eval()
    m.load_state_dict(torch.load(ckpt, map_location="cuda"))
    return m, ns["VOCAB"], ns["IDX"], ns["encode"]

def fwd(m, ids):
    o = m(ids)
    return o[0] if isinstance(o, tuple) else o

# ---------------- (1) grammatical well-formedness ----------------
def eval_grammar(m, V, IDX, enc, n=500, temp=0.8, seed=0, maxlen=160):
    EOS = IDX["<eos>"]
    torch.manual_seed(seed)
    ok = 0; fails = {}; sents = []
    for _ in range(n):
        ids = torch.tensor([[EOS]], device="cuda"); out = []
        for _ in range(maxlen):
            with torch.no_grad(): lg = fwd(m, ids[:, -256:])[0, -1]
            nxt = torch.multinomial(torch.softmax(lg / temp, -1), 1)
            if nxt.item() == EOS: break
            out.append(nxt.item()); ids = torch.cat([ids, nxt[None]], 1)
        s = "".join(V[i] for i in out).strip()
        sents.append(s)
        good, bad = G.check(s)
        ok += good
        for b in bad: fails[b] = fails.get(b, 0) + 1
    return {"n": n, "ok": ok, "pct": 100*ok/n, "fails": fails, "samples": sents[:5]}

# ---------------- (2) next-word admissibility ----------------
POS_SETS = {
    "noun_sg": G.SG_NOUNS, "noun_pl": G.PL_NOUNS,
    "adj": set(sw.ADJECTIVES), "det": G.ALL_DET,
    "prep": {p[0] for p in sw.PREPOSITIONS},
    "adv": (set(sw.ADVERBS_MANNER) | set(sw.ADVERBS_FREQUENCY)
            | set(sw.ADVERBS_DEGREE) | {a for g in sw.ADVERBS_TIME.values() for a in g}),
    "verb": G.ALL_VERB_FORMS, "aux": G.AUXES,
    "conj": set(sw.COORD_CONJUNCTIONS) | set(sw.SUBORDINATORS),
    "pron": set(G.SUBJ_PRON) | G.OBJ_PRON,
}

def admissible(prefix_words, gold):
    """Word classes that could grammatically follow this prefix.

    Approximated by the class of the gold continuation plus classes that are
    always legal there. Deliberately generous: this measures 'did the model
    pick something of the right kind', not 'did it pick the exact word'.
    """
    allow = set()
    for name, s in POS_SETS.items():
        if gold in s: allow |= s
    if not prefix_words: allow |= POS_SETS["det"] | POS_SETS["pron"] | POS_SETS["noun_pl"]
    last = prefix_words[-1] if prefix_words else ""
    if last in G.ALL_DET: allow |= POS_SETS["adj"] | POS_SETS["noun_sg"] | POS_SETS["noun_pl"]
    if last in sw.ADJECTIVES: allow |= POS_SETS["adj"] | POS_SETS["noun_sg"] | POS_SETS["noun_pl"]
    return allow or set(gold)

def eval_nextword(m, V, IDX, enc, path, n=400, seed=0, maxgen=14):
    EOS = IDX["<eos>"]
    rng = random.Random(seed)
    lines = [l.rstrip() for l in open(path) if l.strip()]
    rng.shuffle(lines)
    exact = adm = tot = 0
    for line in lines[:n]:
        words = re.findall(r"[A-Za-z']+", line)
        if len(words) < 4: continue
        k = rng.randint(2, len(words) - 2)
        # rebuild the prefix from the raw line so spacing/punctuation match
        idx = 0
        for _ in range(k):
            idx = line.index(words[_], idx) + len(words[_])
        prefix = line[:idx] + " "
        gold = words[k]
        ids = torch.tensor([[EOS] + enc(prefix)], device="cuda")
        out = []
        for _ in range(maxgen):
            with torch.no_grad(): lg = fwd(m, ids[:, -256:])[0, -1]
            nxt = int(lg.argmax())                      # greedy
            if nxt == EOS: break
            ch = V[nxt]
            if ch == " " and out: break
            out.append(ch); ids = torch.cat([ids, torch.tensor([[nxt]], device="cuda")], 1)
        pred = "".join(out).strip().strip(".,?")
        tot += 1
        if pred.lower() == gold.lower(): exact += 1
        if pred.lower() in {w.lower() for w in admissible(
                [w.lower() for w in words[:k]], gold.lower())}: adm += 1
    return {"n": tot, "exact": exact, "exact_pct": 100*exact/max(tot,1),
            "admissible": adm, "adm_pct": 100*adm/max(tot,1)}

# ---------------- (3) minimal pairs ----------------
def build_pairs(seed=0, per=120):
    rng = random.Random(seed)
    pairs = []   # (good, bad, phenomenon)
    sg = [n for n in sw.NOUNS if n[0] not in sw.UNCOUNTABLE]
    intrans = [v for v in sw.VERBS if not v["trans"]]
    for _ in range(per):
        s, p, _ = rng.choice(sg); v = rng.choice(intrans)
        t3 = sw.third_person_s(v["base"])
        pairs.append((f"The {s} {t3} well.",  f"The {s} {v['base']} well.",  "agree_sv"))
        pairs.append((f"The {p} {v['base']} well.", f"The {p} {t3} well.",   "agree_sv"))
        pairs.append((f"These {p} {v['base']} well.", f"These {s} {v['base']} well.", "det_num"))
        pairs.append((f"This {s} {t3} well.",  f"This {p} {t3} well.",       "det_num"))
        pairs.append((f"The {s} is {v['ing']} well.", f"The {s} is {v['base']} well.", "aux_form"))
        pairs.append((f"The {s} has {v['pp']} well.", f"The {s} has {v['ing']} well.", "aux_form"))
        pairs.append((f"The {s} will {v['base']} well.", f"The {s} will {t3} well.",  "modal"))
        pairs.append((f"The {s} didn't {v['base']} well.", f"The {s} didn't {v['past']} well.", "negation")) 
    rng.shuffle(pairs)
    return pairs

def nll(m, IDX, enc, s):
    EOS = IDX["<eos>"]
    ids = torch.tensor([[EOS] + enc(s)], device="cuda")
    with torch.no_grad(): lg = fwd(m, ids)
    lp = torch.log_softmax(lg[0, :-1].float(), -1)
    tgt = ids[0, 1:]
    return -lp.gather(1, tgt[:, None]).mean().item()   # per-token, length-normalized

def eval_minpairs(m, V, IDX, enc, seed=0, per=120):
    pairs = build_pairs(seed, per)
    by = {}
    for good, bad, phen in pairs:
        win = nll(m, IDX, enc, good) < nll(m, IDX, enc, bad)
        d = by.setdefault(phen, [0, 0]); d[0] += win; d[1] += 1
    out = {k: {"correct": v[0], "n": v[1], "pct": 100*v[0]/v[1]} for k, v in by.items()}
    tc = sum(v[0] for v in by.values()); tn = sum(v[1] for v in by.values())
    out["OVERALL"] = {"correct": tc, "n": tn, "pct": 100*tc/tn}
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt"); ap.add_argument("--arch", required=True)
    ap.add_argument("--eval-file", default="eng_eval_1000tpp.txt")
    ap.add_argument("--n-gen", type=int, default=500)
    ap.add_argument("--n-next", type=int, default=400)
    ap.add_argument("--per-pair", type=int, default=120)
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    m, V, IDX, enc = load(a.arch, a.ckpt)
    print(f"### {a.label or a.ckpt}")
    g = eval_grammar(m, V, IDX, enc, n=a.n_gen)
    print(f"[1] grammar    {g['ok']}/{g['n']} = {g['pct']:.2f}%   fails={g['fails']}")
    w = eval_nextword(m, V, IDX, enc, a.eval_file, n=a.n_next)
    print(f"[2] next-word  exact {w['exact_pct']:.2f}%  admissible {w['adm_pct']:.2f}%  (n={w['n']})")
    p = eval_minpairs(m, V, IDX, enc, per=a.per_pair)
    for k in sorted(p):
        if k != "OVERALL": print(f"[3] {k:12} {p[k]['pct']:6.2f}%  ({p[k]['correct']}/{p[k]['n']})")
    print(f"[3] {'OVERALL':12} {p['OVERALL']['pct']:6.2f}%  ({p['OVERALL']['correct']}/{p['OVERALL']['n']})")
    import json; print("JSON " + json.dumps({"label": a.label, "grammar": g, "nextword": w, "minpairs": p}))
