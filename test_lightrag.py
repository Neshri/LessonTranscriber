#!/usr/bin/env python3
"""
Standalone external test for LightRAG + Ollama integration.

This script bypasses the full LessonTranscriber pipeline and tests
LightRAG directly so we can validate:
  1. LightRAG initializes correctly with our Ollama LLM + embedding setup
  2. Documents can be inserted (ainsert)
  3. Queries return sensible results in all supported modes
  4. The JSON output format we expect actually works

Usage:
    python test_lightrag.py

Prerequisites:
    - Ollama is running at http://127.0.0.1:11434
    - The LLM model (gpt-oss:20b) is pulled in Ollama
    - The embedding model (nomic-embed-text) is pulled in Ollama
    - lightrag-hku is installed  (pip install lightrag-hku)
"""

import asyncio
import json
import logging
import os
import shutil
import sys
import numpy as np
from pathlib import Path

# ─── Configuration (mirrors config.json) ────────────────────────────────────
OLLAMA_URL        = "http://127.0.0.1:11434"
LLM_MODEL         = "gpt-oss:20b"
EMBEDDING_MODEL   = "nomic-embed-text"
EMBEDDING_DIM     = 768          # nomic-embed-text output dimension
WORKING_DIR       = "./test_lightrag_storage"

# ─── Logging setup ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("test_lightrag")

# ─── Sample transcript (short Swedish IT-lesson excerpt) ─────────────────────
SAMPLE_TRANSCRIPT = """
Okej, idag ska vi prata om subnetting och vad det innebär i ett IP-nätverk.
En IP-adress består av 32 bitar och är uppdelad i en nätverksdel och en hostdel.
Vi använder en nätmask, till exempel 255.255.255.0, för att särskilja dessa delar.
Det kallas också för /24 i CIDR-notation, vilket betyder att de 24 första bitarna är nätverket.

Om vi har en /24 kan vi ha 254 användbara hostar, eftersom .0 är nätverksadressen
och .255 är broadcast-adressen.

VLSM, Variable Length Subnet Masking, låter oss dela upp ett nät i mindre bitar.
Det är viktigt när man designar nätverk och vill undvika att slösa IP-adresser.
Till exempel kan ett /24-nät delas upp i flera /26-nät för olika avdelningar.

Router behöver känna till alla subnät för att kunna dirigera trafik korrekt.
Det sköts antingen med statisk routing eller dynamiska routingprotokoll som OSPF eller EIGRP.
"""

QUERY_JSON = (
    "Skapa en tekniskt korrekt, kortfattad och språkligt flytande sammanfattning av lektionen. "
    "Fokusera på vad som faktiskt förklarades verbalt. "
    "Svara ENDAST med ett JSON-objekt enligt detta format exakt: "
    '{"subject": "Ämnesrad", "summary": "- Punkt 1\\n- Punkt 2"}'
)

QUERY_PLAIN = "Vad handlade lektionen om? Besvara på svenska."


# ─── Helpers ─────────────────────────────────────────────────────────────────

def print_section(title: str):
    logger.info("=" * 60)
    logger.info(f"  {title}")
    logger.info("=" * 60)


def cleanup_working_dir():
    if Path(WORKING_DIR).exists():
        shutil.rmtree(WORKING_DIR)
        logger.info(f"Cleaned up working dir: {WORKING_DIR}")
    Path(WORKING_DIR).mkdir(parents=True, exist_ok=True)


# ─── Main async test ─────────────────────────────────────────────────────────

async def run_test():
    # Late import so the script still gives a helpful error if not installed
    try:
        from lightrag import LightRAG, QueryParam
        from lightrag.llm.ollama import ollama_model_complete, ollama_embed
        from lightrag.utils import wrap_embedding_func_with_attrs
    except ImportError as e:
        logger.error(f"Import failed: {e}")
        logger.error("Make sure lightrag-hku is installed:  pip install lightrag-hku")
        sys.exit(1)

    # ── Step 1: clean slate ───────────────────────────────────────────────
    print_section("Step 1: Preparing working directory")
    cleanup_working_dir()

    # ── Step 2: build embedding function ─────────────────────────────────
    print_section("Step 2: Building embedding function")

    @wrap_embedding_func_with_attrs(
        embedding_dim=EMBEDDING_DIM,
        max_token_size=8192,
        model_name=EMBEDDING_MODEL,
    )
    async def embedding_func(texts: list[str]) -> np.ndarray:
        return await ollama_embed.func(
            texts,
            embed_model=EMBEDDING_MODEL,
            host=OLLAMA_URL,
        )

    logger.info(f"Embedding model : {EMBEDDING_MODEL}")
    logger.info(f"Embedding dim   : {EMBEDDING_DIM}")

    # ── Step 3: create LightRAG instance ─────────────────────────────────
    print_section("Step 3: Creating LightRAG instance")

    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=ollama_model_complete,
        llm_model_name=LLM_MODEL,
        embedding_func=embedding_func,
        enable_llm_cache=False,
        llm_model_kwargs={
            "host": OLLAMA_URL,
            "options": {"num_ctx": 32768},
        },
    )
    logger.info(f"LLM model       : {LLM_MODEL}")
    logger.info(f"Ollama URL      : {OLLAMA_URL}")
    logger.info(f"Working dir     : {WORKING_DIR}")

    # ── Step 4: initialize storages ───────────────────────────────────────
    print_section("Step 4: Initializing LightRAG storages")
    await rag.initialize_storages()
    logger.info("Storages initialized OK")

    # ── Step 5: insert document ───────────────────────────────────────────
    print_section("Step 5: Inserting sample transcript")
    logger.info(f"Transcript length: {len(SAMPLE_TRANSCRIPT)} chars")
    await rag.ainsert(SAMPLE_TRANSCRIPT)
    logger.info("Insert complete")

    # ── Step 6: query in different modes ─────────────────────────────────
    results = {}

    for mode in ["naive", "local", "global", "hybrid"]:
        print_section(f"Step 6-{mode}: Query mode={mode}")
        try:
            answer = await rag.aquery(
                QUERY_PLAIN,
                param=QueryParam(
                    mode=mode,
                    top_k=10,
                    enable_rerank=False,
                ),
            )
            logger.info(f"[{mode}] Response:\n{answer}")
            results[mode] ={"ok": True, "response": answer}
        except Exception as e:
            logger.error(f"[{mode}] FAILED: {e}")
            results[mode] = {"ok": False, "error": str(e)}

    # ── Step 7: JSON query (mirrors production usage) ─────────────────────
    print_section("Step 7: JSON-format query (hybrid mode, production-style)")
    try:
        json_answer = await rag.aquery(
            QUERY_JSON,
            param=QueryParam(
                mode="hybrid",
                top_k=20,
                response_type="Single JSON object",
                enable_rerank=False,
            ),
        )
        logger.info(f"Raw JSON response:\n{json_answer}")

        # Try to parse it
        try:
            parsed = json.loads(json_answer)
            logger.info(f"✅  JSON parsed OK!")
            logger.info(f"    subject : {parsed.get('subject')}")
            logger.info(f"    summary : {parsed.get('summary')[:200]}...")
            results["json_query"] = {"ok": True, "parsed": parsed}
        except json.JSONDecodeError as je:
            logger.warning(f"⚠️  JSON parse failed ({je}). Raw output saved.")
            results["json_query"] = {"ok": False, "raw": json_answer, "parse_error": str(je)}
    except Exception as e:
        logger.error(f"JSON query FAILED: {e}")
        results["json_query"] = {"ok": False, "error": str(e)}

    # ── Summary ───────────────────────────────────────────────────────────
    print_section("Test Summary")
    all_ok = all(v["ok"] for v in results.values())
    for step, res in results.items():
        status = "✅ OK" if res["ok"] else "❌ FAIL"
        logger.info(f"  {step:12s} {status}")

    if all_ok:
        logger.info("\n✅  All steps passed!")
    else:
        logger.warning("\n⚠️  Some steps failed — see output above for details.")

    return results


if __name__ == "__main__":
    asyncio.run(run_test())
