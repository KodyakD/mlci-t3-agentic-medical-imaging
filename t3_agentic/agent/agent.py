"""The agent loop.

    gate modal (deterministe)
      -> proposition d'outils par le LLM (bornee aux outils autorises)
      -> execution (arguments injectes par le harnais)
      -> OBSERVATION (les resultats sont renvoyes au LLM)
      -> decision suivante (nouvel outil possible) -- boucle bornee
      -> completion harnais (outils requis manquants)
      -> fusion par regles -> reponse (avec controle de provenance)

Le LLM ne decide jamais d'un label, d'un score ou d'un refus.
"""
import json
import os
import re
import urllib.request
import uuid

from . import logger
from .calibration import load_config
from .fusion import fuse
from .gate import (allowed_tools, complete_tools, fallback_route, required_tools,
                   validate_tool_choice)
from .mcp_client import MCPPool
from .prompts import EXPLAIN_SYSTEM, SYSTEM, TOOL_SCHEMAS, WARNING

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")
MAX_ROUNDS = int(os.environ.get("T3_MAX_ROUNDS", "3"))
MAX_CALLS = int(os.environ.get("T3_MAX_CALLS", "5"))


def ollama_chat(messages, tools=None, model=OLLAMA_MODEL, timeout=300):
    payload = {"model": model, "messages": messages, "stream": False,
               "options": {"temperature": 0, "num_ctx": 4096}}
    if tools:
        payload["tools"] = tools
    req = urllib.request.Request(
        OLLAMA_URL, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _extract_tool_names(message):
    names = []
    for call in message.get("tool_calls") or []:
        fn = call.get("function") or {}
        if fn.get("name"):
            names.append(fn["name"])
    return names


def _numbers(text):
    # file paths (gradcam_NORMAL_0.png) must not be parsed as numbers
    text = re.sub(r"[\w\-./\\]*\.(?:png|jpe?g|keras|h5|json|txt|csv)", " ", text or "")
    return set(re.findall(r"\d+[.,]?\d*", text))


def _as_float(token):
    try:
        return float(token.replace(",", "."))
    except ValueError:
        return None


def _warning_ok(text):
    import unicodedata
    norm = unicodedata.normalize("NFKD", text or "")
    norm = "".join(c for c in norm if not unicodedata.combining(c)).lower()
    return "prototype" in norm and "diagnostic medical" in norm


def _provenance_ok(answer, allowed_text):
    """Every number in the answer must trace back to a tool result."""
    allowed = [v for v in (_as_float(n) for n in _numbers(allowed_text)) if v is not None]
    for token in _numbers(answer):
        v = _as_float(token)
        if v is None:
            continue
        candidates = [v] + ([v / 100.0] if v > 1.0 else [])
        if not any(abs(c - a) <= 0.01 for c in candidates for a in allowed):
            return False
    return True


def _tool_args(tool, image_path, note, already_preprocessed, question=""):
    if tool == "classify_image":
        return {"path": image_path, "already_preprocessed": already_preprocessed}
    if tool == "explain_image":
        return {"path": image_path, "already_preprocessed": already_preprocessed}
    if tool == "analyze_text":
        return {"note": note or ""}
    if tool == "get_guideline":
        return {"topic": ""}
    if tool == "search_guidelines":
        return {"query": question, "k": 3}
    return {}


def _execute(tool, trace, pool, image_path, note, already_preprocessed, question=""):
    args = _tool_args(tool, image_path, note, already_preprocessed, question)
    key = (tool, json.dumps(args, sort_keys=True))
    if key in trace.setdefault("_seen_calls", set()):
        return None, True  # useless loop: identical call already made
    trace["_seen_calls"].add(key)
    try:
        call = pool.call_tool(tool, args)
        payload, duration = call["result"], call["duration_s"]
    except Exception as e:
        payload, duration = {"error": str(e)[:300]}, 0.0
        trace["failures"] = trace.get("failures", 0) + 1
    logger.log_tool_call(trace, tool, args, payload, duration,
                         forced=trace.get("_forced_now", False))
    trace["tool_calls"].append({"tool": tool, "duration_s": duration,
                                "error": "error" in payload})
    return payload, False


def _template_answer(fusion, vision, text):
    label_fr = {"pneumonie": "pneumonie", "normal": "normal"}
    if fusion["refused"]:
        return (f"Je ne conclus pas : {fusion['reason']}. "
                f"Confiance finale : {fusion['confidence']}. {WARNING}")
    vi = vision.get("label") if vision else None
    sc = vision.get("score") if vision else None
    parts = []
    if vision and vi:
        parts.append(f"Image : suspicion de {label_fr.get(vi, vi)} "
                     f"(score {sc:.2f})." if isinstance(sc, float)
                     else f"Image : suspicion de {label_fr.get(vi, vi)}.")
    if text:
        parts.append(f"Note clinique : {text.get('orientation')} "
                     f"(suffisant={text.get('suffisant')}).")
    parts.append(f"Coherence : {fusion['coherence']} (confiance {fusion['confidence']}).")
    parts.append(f"Conclusion (suspicion, non diagnostic) : {fusion['final']}.")
    parts.append(WARNING)
    return " ".join(parts)


def run_agent(user_msg, image_path=None, note=None, pool=None,
              already_preprocessed=False, use_llm=True, max_rounds=MAX_ROUNDS,
              use_critic=True):
    """Run one agent turn. pool: an open MCPPool (created/closed by the caller)."""
    trace = {
        "run_id": uuid.uuid4().hex[:8],
        "question": user_msg,
        "modalities": {"image": bool(image_path), "text": bool(note)},
        "tool_calls": [],
        "fusion": None,
        "refusal": None,
        "tool_choice_source": None,
        "provenance_violation": False,
        "rounds": 0,
        "failures": 0,
        "useless_loops": 0,
        "_seen_calls": set(),
    }
    allowed = allowed_tools(bool(image_path), note is not None)
    required = required_tools(bool(image_path), note is not None)
    if not allowed:
        answer = ("Aucune modalite exploitable et aucune ressource applicable. " + WARNING)
        trace["answer"] = answer
        logger.log_run(trace)
        return {"answer": answer, "fusion": None, "trace": trace}

    schemas = [t for t in TOOL_SCHEMAS if t["function"]["name"] in allowed]
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": user_msg}]
    executed = {}
    fallback_used = False
    proposed_all = []
    forced = []

    # ---------------- observation -> next decision loop (bounded) ----------------
    for rnd in range(max_rounds):
        trace["rounds"] = rnd + 1
        proposed = []
        if use_llm:
            try:
                msg = ollama_chat(messages, tools=schemas)
                proposed = validate_tool_choice(_extract_tool_names(msg.get("message", {})), allowed)
                trace["tool_choice_source"] = "llm" if proposed else "fallback"
                if not proposed:
                    fallback_used = True
            except Exception as e:
                fallback_used = True
                trace["llm_error"] = str(e)[:200]
                trace["tool_choice_source"] = "fallback"
        else:
            fallback_used = True
            trace["tool_choice_source"] = "rule_router"

        if not proposed:
            proposed = fallback_route(user_msg, bool(image_path), note is not None)
            proposed = [t for t in proposed if t in allowed]
            fallback_used = True
            trace["tool_choice_source"] = "rule_router"

        # only tools not yet executed are useful
        proposed = [t for t in proposed if t not in executed]
        if not proposed:
            break
        proposed_all += proposed

        for tool in proposed:
            if len(executed) >= MAX_CALLS:
                break
            payload, useless = _execute(tool, trace, pool, image_path, note,
                                        already_preprocessed, user_msg)
            if useless:
                trace["useless_loops"] += 1
                continue
            if payload is not None:
                executed[tool] = payload

        # OBSERVATION -> the next round sees what the tools returned
        messages.append({"role": "user",
                         "content": "Observations des outils : " +
                                    json.dumps(executed, ensure_ascii=False)})

    # ---------------- harness completion: required tools must have run ----------
    for tool in complete_tools(list(executed.keys()), required):
        if len(executed) >= MAX_CALLS:
            break
        trace["_forced_now"] = True
        forced.append(tool)
        payload, _ = _execute(tool, trace, pool, image_path, note,
                              already_preprocessed, user_msg)
        if payload is not None:
            executed[tool] = payload

    # ---------------- step 4 of the brief: contextualise with the resource ------
    # Cheap, deterministic, always useful: the guidelines are consulted even when
    # the LLM never proposed them (they contextualise the answer; they never
    # change the fusion decision).
    if (image_path or note) and "get_guideline" not in executed:
        trace["_forced_now"] = True
        payload, _ = _execute("get_guideline", trace, pool, image_path, note,
                              already_preprocessed, user_msg)
        if payload is not None:
            executed["get_guideline"] = payload
            forced.append("get_guideline")
    trace["_forced_now"] = False
    trace["proposed"] = proposed_all
    trace["forced"] = forced

    vision = executed.get("classify_image")
    text = executed.get("analyze_text")
    resource = executed.get("get_guideline") or executed.get("search_guidelines")

    def composer_facts(vis):
        """Facts for the final-wording LLM: calibration artifacts (temperature,
        TAU thresholds) are model internals, a patient vital they are not. The
        model has to never be tempted to report them."""
        if not isinstance(vis, dict):
            return vis
        facts = {k: v for k, v in vis.items()
                 if k not in ("temperature", "calibrated", "model", "patient")}
        return facts
    if vision and executed.get("explain_image") and "error" not in executed["explain_image"]:
        vision = dict(vision)
        vision["suspect"] = executed["explain_image"].get("suspect")
        vision["inside_ratio"] = executed["explain_image"].get("inside_ratio")
        vision["heatmap"] = executed["explain_image"].get("heatmap")

    # ---------------- fusion (only if a modality produced something) ----------
    if vision or text:
        fusion = fuse(vision, text)
        trace["fusion"] = fusion
        trace["refusal"] = fusion["refused"]
        logger.log_fusion(trace, fusion)
        allowed_text = json.dumps(
            {"fusion": fusion, "image": composer_facts(vision),
             "texte": text,
             "reperes_guidelines": (resource.get("result", resource)
                                    if isinstance(resource, dict) else resource)},
            ensure_ascii=False)
        answer = _template_answer(fusion, vision, text)
    elif resource:
        # knowledge question: the answer comes from the MCP resource
        fusion = {"coherence": "ressource", "final": None,
                  "reason": "reponse fondee sur guidelines://pneumonia",
                  "confidence": "moderee", "refused": False, "warning": WARNING}
        trace["fusion"] = fusion
        logger.log_fusion(trace, fusion)
        guidelines = resource.get("result", resource) if isinstance(resource, dict) else resource
        if isinstance(guidelines, list):  # search_guidelines returns passages
            guidelines = " ".join(
                f"- {h.get('title', '')} : {h.get('text', '')}" for h in guidelines
                if isinstance(h, dict) and "error" not in h)
        if not isinstance(guidelines, str) or "error" in guidelines or not guidelines.strip():
            # the MCP resource call failed: fall back to the same content in-process,
            # a transport error must never reach the user
            from ..resources.content import GUIDELINES_PNEUMONIA
            guidelines = GUIDELINES_PNEUMONIA
            trace["resource_fallback"] = True
        allowed_text = json.dumps({"ressource": guidelines}, ensure_ascii=False)
        # deterministic fallback wording (used when the LLM is unavailable)
        answer = ("Voici les reperes de lecture d'une radiographie thoracique : "
                  + str(guidelines).strip()[:600] + " " + WARNING)
    else:
        fusion = {"coherence": "insuffisant", "final": None,
                  "reason": "aucun outil exploitable", "confidence": "aucune",
                  "refused": True, "warning": WARNING}
        trace["fusion"] = fusion
        trace["refusal"] = True
        logger.log_fusion(trace, fusion)
        allowed_text = json.dumps({"fusion": fusion}, ensure_ascii=False)
        answer = _template_answer(fusion, None, None)

    # ---------------- final wording by the LLM, provenance-checked ------------
    if use_llm and (vision or text or resource):
        try:
            msg = ollama_chat([{"role": "system", "content": EXPLAIN_SYSTEM},
                               {"role": "user", "content": allowed_text}])
            candidate = (msg.get("message", {}) or {}).get("content", "").strip()
            if candidate and _provenance_ok(candidate, allowed_text):
                stripped = re.sub(r"[^.]*prototype\s+p[ée]dagogique[^.]*\.", "", candidate,
                                  flags=re.I).strip()
                answer = (stripped + " " + WARNING).strip()
            elif candidate:
                trace["provenance_violation"] = True
                trace["llm_candidate"] = candidate[:500]
        except Exception as e:
            trace["llm_error"] = str(e)[:200]

    # ---------------- critic pass: a second LLM reviews the drafted answer ---
    if use_critic and use_llm and (vision or text or resource):
        from .critic import critique, apply_critic
        facts = {"fusion": fusion, "image": composer_facts(vision), "texte": text}
        if resource and not (vision or text):
            facts = {"fusion": fusion, "ressource": resource}
        try:
            verdict, revised, issues = critique(answer, facts, user_msg)
        except Exception as e:
            verdict, revised, issues = "ok", "", [f"critic error: {str(e)[:120]}"]
        trace["critic_verdict"] = verdict
        trace["critic_issues"] = issues
        if verdict in ("revise", "reject"):
            template = _template_answer(fusion, vision, text)
            if resource and not (vision or text):
                template = answer
            final, used_template = apply_critic(verdict, revised, answer, template, facts)
            if final != answer:
                trace["critic_modified"] = True
            answer = final
            if used_template:
                trace["critic_template_fallback"] = True

    trace.pop("_seen_calls", None)
    trace["answer"] = answer
    logger.log_run(trace)
    return {"answer": answer, "fusion": fusion, "vision": vision, "text": text,
            "resource": resource, "trace": trace}


def _main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Agent T3 (une question par execution)")
    ap.add_argument("ask")
    ap.add_argument("--image")
    ap.add_argument("--note")
    ap.add_argument("--preprocessed", action="store_true",
                    help="l'image a deja passe par preprocess_gray")
    ap.add_argument("--no-llm", action="store_true", help="routage deterministe seul")
    args = ap.parse_args(argv)
    pool = MCPPool()
    try:
        out = run_agent(args.ask, image_path=args.image, note=args.note, pool=pool,
                        already_preprocessed=args.preprocessed, use_llm=not args.no_llm)
    finally:
        pool.close()
    print(json.dumps({k: v for k, v in out.items() if k != "trace"},
                     ensure_ascii=False, indent=2))
    print("\n--- trace ---")
    print(json.dumps(out["trace"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _main()
