"""
Silver Guard — Indian SMS Scam Detector
========================================
Fine-tuned MobileBERT exported as ONNX.

The model output 'threat_score' is already a softmax probability:
    0.0 = definitely safe (ham)
    1.0 = definitely scam

Input format:  HEADER [SEP] message body
  • HEADER  = DLT sender ID (e.g. JD-SBINOT) or phone number (+919876543210)
  • Message = the SMS body text

Examples:
    JD-SBINOT [SEP] Your a/c XX5678 credited with Rs 25,000 by NEFT. -SBI
    +919876543210 [SEP] Your Aadhaar is linked to a money laundering case. Call immediately.
    Hey, are we meeting for dinner tonight?

Usage:
    python silver_guard.py
    python silver_guard.py --model silver_guard.onnx --vocab vocab.txt
"""

import argparse
import os
import re
import sys
import unicodedata
import numpy as np
import onnxruntime as ort

# Fix emoji printing on Windows terminals (cp1252 etc.)
if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# ─────────────────────────────────────────────────────────────────────────────
# Lightweight BERT WordPiece Tokenizer (no transformers library needed)
# Mirrors google/mobilebert-uncased tokenization exactly
# ─────────────────────────────────────────────────────────────────────────────

class WordpieceTokenizer:

    def __init__(self, vocab_file: str, do_lower_case: bool = True):
        self.do_lower_case = do_lower_case
        self.vocab: dict[str, int] = {}

        with open(vocab_file, encoding="utf-8") as f:
            for idx, line in enumerate(f):
                token = line.rstrip("\n")
                self.vocab[token] = idx

        self.unk_id = self.vocab.get("[UNK]", 100)
        self.cls_id = self.vocab.get("[CLS]", 101)
        self.sep_id = self.vocab.get("[SEP]", 102)
        self.pad_id = self.vocab.get("[PAD]", 0)

    # ── Normalisation ──────────────────────────────────────────────────────────

    @staticmethod
    def _clean(text: str) -> str:
        out = []
        for ch in text:
            cp = ord(ch)
            if cp == 0 or cp == 0xFFFD or unicodedata.category(ch) == "Cc":
                continue
            out.append(" " if unicodedata.category(ch) == "Zs" or ch == "\t" else ch)
        return "".join(out)

    @staticmethod
    def _strip_accents(text: str) -> str:
        text = unicodedata.normalize("NFD", text)
        return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")

    def _basic_tokenize(self, text: str) -> list[str]:
        text = self._clean(text)
        if self.do_lower_case:
            text = text.lower()
            text = self._strip_accents(text)
        tokens, buf = [], []
        for ch in text:
            is_punct = (unicodedata.category(ch).startswith("P") or
                        ch in r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~""")
            if ch == " ":
                if buf:
                    tokens.append("".join(buf))
                    buf = []
            elif is_punct:
                if buf:
                    tokens.append("".join(buf))
                    buf = []
                tokens.append(ch)
            else:
                buf.append(ch)
        if buf:
            tokens.append("".join(buf))
        return tokens

    def _wordpiece(self, word: str) -> list[int]:
        if len(word) > 200:
            return [self.unk_id]
        ids, start = [], 0
        while start < len(word):
            end, cur = len(word), None
            while start < end:
                substr = ("##" if start > 0 else "") + word[start:end]
                if substr in self.vocab:
                    cur = self.vocab[substr]
                    break
                end -= 1
            if cur is None:
                return [self.unk_id]
            ids.append(cur)
            start = end
        return ids

    # ── Public API ────────────────────────────────────────────────────────────

    def encode(self, text_a: str, text_b: str | None = None,
               max_length: int = 128) -> dict[str, list[int]]:
        """
        Produces [CLS] tokens_a [SEP] (tokens_b [SEP]) padded to max_length.
        Returns input_ids and attention_mask.
        """
        ids_a = [wp for tok in self._basic_tokenize(text_a)
                 for wp in self._wordpiece(tok)]
        ids_b = ([wp for tok in self._basic_tokenize(text_b)
                  for wp in self._wordpiece(tok)] if text_b else None)

        specials = 3 if ids_b is not None else 2
        budget = max_length - specials

        if ids_b is not None:
            while len(ids_a) + len(ids_b) > budget:
                (ids_a if len(ids_a) >= len(ids_b) else ids_b).pop()
        else:
            ids_a = ids_a[:budget]

        token_ids = (
            [self.cls_id] + ids_a + [self.sep_id]
            + (ids_b + [self.sep_id] if ids_b is not None else [])
        )
        attention = [1] * len(token_ids)

        pad_len     = max_length - len(token_ids)
        token_ids  += [self.pad_id] * pad_len
        attention   += [0]          * pad_len

        return {"input_ids": token_ids, "attention_mask": attention}


# ─────────────────────────────────────────────────────────────────────────────
# DLT Header Analysis  (TRAI sender ID heuristics)
# ─────────────────────────────────────────────────────────────────────────────

def analyse_header(header: str) -> tuple[str, str]:
    """
    Returns (description, emoji) based on the sender ID format.

    Legitimate DLT format:  XX-YYYYYYZ
      Z suffix → G=govt  T=transactional  S=service  P=promotional
    Scam patterns: raw phone numbers, gibberish IDs, wrong suffix
    """
    h = header.strip()
    if not h:
        return "personal / no sender ID", "👤"

    # Raw Indian phone number — extremely common in scam SMS
    if re.match(r"^\+?91\s?\d{10}$", h.replace(" ", "")):
        return "phone number  ⚠️ scam indicator", "📵"
    if re.match(r"^\d{10,13}$", h.replace(" ", "")):
        return "numeric sender  ⚠️ scam indicator", "📵"

    # Proper DLT format: 2 uppercase letters, hyphen, 6-8 alphanumeric
    if re.match(r"^[A-Z]{2}-[A-Z0-9]{6,8}$", h):
        suffix_map = {
            "G": ("government entity ✅",       "🏛️"),
            "T": ("transactional / bank ✅",    "🏦"),
            "S": ("service / OTP ✅",           "⚙️"),
            "P": ("promotional ✅",              "📢"),
        }
        return suffix_map.get(h[-1].upper(), ("registered DLT ✅", "✅"))

    return "unregistered / suspicious ⚠️", "⚠️"


# ─────────────────────────────────────────────────────────────────────────────
# Verdict thresholds
# ─────────────────────────────────────────────────────────────────────────────

THRESHOLDS = [
    (0.80, "🚨 HIGH RISK SCAM",  "Strong scam indicators detected."),
    (0.55, "⚠️  LIKELY SCAM",    "Probably a scam — exercise caution."),
    (0.40, "🟡 BORDERLINE",      "Ambiguous — treat with caution."),
    (0.00, "✅ SAFE (HAM)",      "Message looks legitimate."),
]

def get_verdict(score: float) -> tuple[str, str]:
    for threshold, label, note in THRESHOLDS:
        if score >= threshold:
            return label, note
    return THRESHOLDS[-1][1], THRESHOLDS[-1][2]


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(session: ort.InferenceSession,
                  tokenizer: WordpieceTokenizer,
                  header: str,
                  message: str,
                  max_length: int = 128) -> None:
    """Tokenize → run ONNX → pretty-print result."""

    # Training format: "HEADER [SEP] message"
    # We pass header as text_a and message as text_b to get:
    #   [CLS] header_tokens [SEP] message_tokens [SEP] [PAD]...
    if header:
        encoded = tokenizer.encode(header, message, max_length=max_length)
    else:
        encoded = tokenizer.encode(message, max_length=max_length)

    input_ids      = np.array([encoded["input_ids"]],      dtype=np.int64)
    attention_mask = np.array([encoded["attention_mask"]], dtype=np.int64)

    # ── IMPORTANT: output is already softmax scam probability [0.0 – 1.0]
    # The ONNX wrapper (Cell 6 of training notebook) does:
    #   probs = F.softmax(logits, dim=-1)
    #   return probs[:, 1:2]   ← just the scam probability
    # So we must NOT apply softmax again.
    result = session.run(None, {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
    })
    threat_score = float(np.array(result[0]).flatten()[0])

    label, note          = get_verdict(threat_score)
    hdr_desc, hdr_emoji  = analyse_header(header)

    # ── Visual bar ────────────────────────────────────────────────────────
    filled  = round(threat_score * 30)
    bar     = "█" * filled + "░" * (30 - filled)

    print()
    print("┌" + "─" * 62 + "┐")
    print(f"│  Verdict      : {label:<44}│")
    print(f"│  Threat Score : {threat_score:.4f}  [{bar}] │")
    print(f"│  Note         : {note:<44}│")
    print("├" + "─" * 62 + "┤")
    print(f"│  Sender ID    : {(hdr_emoji + '  ' + hdr_desc):<44}│")
    print("└" + "─" * 62 + "┘")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Silver Guard — Indian SMS scam classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Input format:   HEADER [SEP] message body
Examples:
  JD-SBINOT [SEP] Your a/c XX5678 credited with Rs 25,000. -SBI
  +919876543210 [SEP] Your Aadhaar is linked to money laundering. Call now!
  Hey bro, dinner tonight?
        """
    )
    p.add_argument("--model",      default="silver_guard.onnx",
                   help="Path to ONNX model file (default: silver_guard.onnx)")
    p.add_argument("--vocab",      default="vocab.txt",
                   help="Path to BERT vocab.txt (default: vocab.txt)")
    p.add_argument("--max-length", type=int, default=128,
                   help="Max token sequence length (default: 128)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Validate required files
    for path in (args.model, args.vocab):
        if not os.path.exists(path):
            print(f"❌  File not found: {path}")
            return

    print(f"Loading Silver Guard ({args.model})...")
    try:
        session = ort.InferenceSession(args.model)
    except Exception as e:
        print(f"❌  Could not load model:\n    {e}")
        return

    tokenizer = WordpieceTokenizer(args.vocab, do_lower_case=True)
    print("✅  Ready.\n")

    print("─" * 64)
    print("  Format :  HEADER [SEP] message body")
    print("  Example:  JD-SBINOT [SEP] Your a/c XX5678 credited with Rs 25,000.")
    print("  No DLT?:  Just type the message — no [SEP] needed.")
    print("  Quit   :  type 'quit'")
    print("─" * 64)

    while True:
        try:
            raw = input("\n>>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not raw:
            continue
        if raw.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if "[SEP]" in raw:
            parts   = raw.split("[SEP]", maxsplit=1)
            header  = parts[0].strip()
            message = parts[1].strip()
        else:
            header  = ""
            message = raw

        if not message:
            print("  ⚠️  Please include a message body.")
            continue

        run_inference(session, tokenizer, header, message,
                      max_length=args.max_length)


if __name__ == "__main__":
    main()
