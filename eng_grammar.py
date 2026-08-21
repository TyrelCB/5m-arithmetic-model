"""Grammatical well-formedness checker for synth_word.py output.

Reuses the generator's own word banks, so it measures conformance to that
grammar -- NOT English. A model that perfectly memorized the generator's
quirks scores 100%. That is the right measure for "did this architecture
learn the training distribution", and the wrong one for "does it know English".

Checks, in order of how cheaply they fail:
  lexicon   every token is a known word form
  det_num   determiner/noun number agreement (this dogs, many dog)
  agree     subject-verb agreement (he walk, they walks, I is)
  aux       auxiliary + main-verb form (is walked, has walking, will walked)
  neg       negation form (doesn't walked, didn't ran)
  subj      every finite clause has a subject (After have never found ...)
"""
import re, synth_word as sw

SUBJ_PRON = {"i":"i","you":"pl","he":"hst","she":"hst","it":"hst","we":"pl","they":"pl"}
OBJ_PRON  = {"me","him","her","us","them","you","it"}
POSS      = {"my","your","his","her","its","our","their"}

SG_NOUNS = {n[0] for n in sw.NOUNS}
PL_NOUNS = {n[1] for n in sw.NOUNS}
BASE   = {v["base"]: v for v in sw.VERBS}
PAST   = {v["past"]: v for v in sw.VERBS}
ING    = {v["ing"]: v for v in sw.VERBS}
PP     = {v["pp"]: v for v in sw.VERBS}
THIRD  = {sw.third_person_s(v["base"]): v for v in sw.VERBS}
ALL_VERB_FORMS = set(BASE) | set(PAST) | set(ING) | set(PP) | set(THIRD)

BE      = {"am":"i","is":"hst","are":"pl","was":("i","hst"),"were":"pl"}
HAVE    = {"have":("i","pl"),"has":"hst","had":None}
NEG_DO  = {"don't":("i","pl"),"doesn't":"hst","didn't":None}
NEG_BE  = {"isn't":"hst","aren't":"pl","wasn't":("i","hst"),"weren't":"pl"}
DO_SUP  = {"do":("i","pl"),"does":"hst","did":None}
AUXES   = set(BE)|set(HAVE)|set(sw.MODALS)|set(NEG_DO)|set(NEG_BE)|set(DO_SUP)|{"not"}

DETS_SG = {"a","an","the","this","that","every","each"} | POSS
DETS_PL = {"the","these","those"} | set(sw.QUANTIFIERS_PL) | POSS
ALL_DET = DETS_SG | DETS_PL

LEXICON = (SG_NOUNS | PL_NOUNS | ALL_VERB_FORMS | AUXES | ALL_DET
           | set(SUBJ_PRON) | OBJ_PRON | set(sw.ADJECTIVES)
           | set(sw.ADVERBS_MANNER) | set(sw.ADVERBS_FREQUENCY)
           | set(sw.ADVERBS_DEGREE) | {a for g in sw.ADVERBS_TIME.values() for a in g}
           | {p[0] for p in sw.PREPOSITIONS} | set(sw.COORD_CONJUNCTIONS)
           | set(sw.SUBORDINATORS) | {w.lower() for w in sw.WH_SUBJECT+sw.WH_OTHER})

def toks(s):
    return re.findall(r"[a-z']+", s.lower())

CLAUSE_EDGE = set(sw.COORD_CONJUNCTIONS) | set(sw.SUBORDINATORS)

def person_of(words, i):
    """Person code for the subject of THIS clause, or None.

    Stops at clause boundaries -- scanning past a conjunction picks up the
    previous clause's subject and produces false agreement errors on every
    multi-clause sentence.
    """
    for j in range(i-1, -1, -1):
        w = words[j]
        if w in CLAUSE_EDGE: return None
        if w in SUBJ_PRON: return SUBJ_PRON[w]
        if w in PL_NOUNS:  return "pl"
        if w in SG_NOUNS:  return "hst"
    return None

WH = {w.lower() for w in sw.WH_SUBJECT + sw.WH_OTHER}

def is_question(sent, w):
    """Questions use inverted order; the statement-order checks do not apply."""
    return sent.rstrip().endswith("?") or (w and (w[0] in WH or w[0] in AUXES))

def check(sent):
    """Return (ok, [failed_check_names])."""
    w = toks(sent)
    if not w: return False, ["empty"]
    bad = []
    q = is_question(sent, w)

    # --- lexicon ---
    unknown = [t for t in w if t not in LEXICON]
    if unknown: bad.append("lexicon")

    # --- determiner / number agreement ---
    for i, t in enumerate(w):
        if t in ALL_DET and i+1 < len(w):
            # skip adjectives/degree adverbs to reach the head noun
            j = i+1
            while j < len(w) and (w[j] in sw.ADJECTIVES or w[j] in sw.ADVERBS_DEGREE): j += 1
            if j >= len(w): continue
            head = w[j]
            if head in PL_NOUNS and head not in SG_NOUNS and t in (DETS_SG - DETS_PL):
                bad.append("det_num"); break
            if head in SG_NOUNS and head not in PL_NOUNS and t in (DETS_PL - DETS_SG) \
               and head not in sw.UNCOUNTABLE:
                bad.append("det_num"); break

    # --- subject-verb agreement (present tense + be/have) ---
    for i, t in enumerate(w if not q else []):
        p = person_of(w, i)
        if p is None: continue
        if t in THIRD and t not in BASE and p in ("i","pl"): bad.append("agree"); break
        # a form that is also a past tense / participle is not evidence of a
        # bare-base agreement error ("rice read the chairs" is past tense)
        if (t in BASE and t not in THIRD and t not in PAST and t not in PP
                and p == "hst" and t not in ALL_DET):
            # bare base with 3sg subject is only OK after a modal / do-support
            # walk back past negation and adverbs to the real licensor
            k = i-1
            while k >= 0 and (w[k] == "not" or w[k] in sw.ADVERBS_FREQUENCY
                              or w[k] in sw.ADVERBS_MANNER or w[k] in sw.ADVERBS_DEGREE):
                k -= 1
            prev = w[k] if k >= 0 else ""
            if prev not in sw.MODALS and prev not in NEG_DO and prev not in DO_SUP and prev != "to":
                bad.append("agree"); break
        for tbl in (BE, HAVE):
            if t in tbl and tbl[t] is not None:
                want = tbl[t]; want = want if isinstance(want, tuple) else (want,)
                if p not in want: bad.append("agree"); break
        else:
            continue
        break

    # --- auxiliary + main verb form ---
    for i, t in enumerate(w[:-1]):
        nxt = w[i+1]
        j = i+1
        while j < len(w) and (w[j] in sw.ADVERBS_FREQUENCY or w[j]=="not"): j += 1
        if j >= len(w): continue
        nxt = w[j]
        if nxt not in ALL_VERB_FORMS: continue
        if t in BE and nxt not in ING and nxt not in PP: bad.append("aux"); break
        if t in HAVE and nxt not in PP: bad.append("aux"); break
        if t in sw.MODALS and nxt not in BASE: bad.append("aux"); break

    # --- negation form ---
    for i, t in enumerate(w[:-1]):
        if t in NEG_DO:
            j = i+1
            while j < len(w) and w[j] in sw.ADVERBS_FREQUENCY: j += 1
            if j < len(w) and w[j] in ALL_VERB_FORMS and w[j] not in BASE:
                bad.append("neg"); break

    # --- subject presence after a subordinator ---
    for i, t in enumerate(w if not q else []):
        if t in sw.SUBORDINATORS and i+1 < len(w):
            nxt = w[i+1]
            if nxt in AUXES or (nxt in ALL_VERB_FORMS and nxt not in ING):
                bad.append("subj"); break

    return (not bad), sorted(set(bad))

if __name__ == "__main__":
    import sys
    good = ["She is drinking these angry cakes.",
            "We had not waited, but it never makes us.",
            "They will usually visit me quietly tomorrow."]
    bad  = ["That sweet child could not close many short cars make that rabbit.",
            "After have never found the black roads, we loved you.",
            "Bread works late, as theyfore each hot car ay talked.",
            "He walk to the school.", "The dogs runs quickly.",
            "She is walked the road.", "He doesn't walked."]
    for s in good: print("EXPECT-OK  ", check(s), s)
    for s in bad:  print("EXPECT-BAD ", check(s), s)
