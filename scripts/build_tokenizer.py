"""Build out/tokenizer: flan-t5-small's SentencePiece model plus pieces for capital umlauts.

The base vocab has no piece for Ä/Ö/ẞ, so they decode as <unk> and vanish ("Ökologie" → "kologie").
They are added as real SentencePiece pieces, not tokenizer added_tokens: Transformers.js inserts a stray
word break after added_tokens ("Ö sterreich"), while real pieces tokenize identically in Python and JS.
"""
import shutil
from pathlib import Path

from sentencepiece import sentencepiece_model_pb2 as sp_pb2
from transformers import AutoTokenizer, T5Tokenizer, T5TokenizerFast

# flan-t5-small/-base/-large ship byte-identical spiece.model files, so this tokenizer fits any of them.
BASE_MODEL = "google/flan-t5-small"
OUT = Path("out/tokenizer")
# "▁"-prefixed variants keep a capitalised word one token longer at most ("▁Ö" + "sterreich")
NEW_PIECES = ["▁Ä", "▁Ö", "Ä", "Ö", "ẞ"]
ROUND_TRIP = ["Ökologie", "Ärger", "Österreich", "ÄRGER", "GROẞ", "Das Ärgernis ist groß.",
              "sprecher: Er | Ich bin in Österreich geboren.", "Ich bin 18 Jahre alt."]


def main():
    base = T5Tokenizer.from_pretrained(BASE_MODEL)
    proto = sp_pb2.ModelProto()
    proto.ParseFromString(Path(base.vocab_file).read_bytes())
    existing = {p.piece for p in proto.pieces}
    # below every existing score, so a new piece never outcompetes an existing multi-character piece
    score = min(p.score for p in proto.pieces if p.type == sp_pb2.ModelProto.SentencePiece.NORMAL) - 1.0
    for text in NEW_PIECES:
        if text not in existing:
            piece = proto.pieces.add()
            piece.piece, piece.score, piece.type = text, score, sp_pb2.ModelProto.SentencePiece.NORMAL

    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True)
    (OUT / "spiece.model").write_bytes(proto.SerializeToString())
    slow = T5Tokenizer(str(OUT / "spiece.model"), extra_ids=100, legacy=False)
    T5TokenizerFast(__slow_tokenizer=slow, vocab_file=str(OUT / "spiece.model"), extra_ids=100, legacy=False,
                    from_slow=True).save_pretrained(OUT)

    tok = AutoTokenizer.from_pretrained(OUT)
    orig = AutoTokenizer.from_pretrained(BASE_MODEL)
    for text in ROUND_TRIP:
        back = tok.decode(tok(text)["input_ids"], skip_special_tokens=True)
        assert back == text, f"round-trip failed: {text!r} -> {back!r}"
        if not any(c in text for c in "ÄÖẞ"):
            # pretrained weights are only reusable if every other string keeps its ids
            assert tok(text)["input_ids"] == orig(text)["input_ids"], f"ids changed for {text!r}"
    print(f"wrote {OUT} ({len(tok)} tokens, {len(proto.pieces)} SentencePiece pieces)")


if __name__ == "__main__":
    main()
