"""Streaming Simple-9 / Elias-Fano readers vs the bulk reference decoders."""
import pytest


def test_codec_readers_match_bulk_decoders():
    try:
        import cpp_engine
    except ImportError:
        try:
            from submission import cpp_engine
        except ImportError:
            pytest.skip("cpp_engine not built")
    cpp_engine.check_codecs()


def test_dotted_acronyms_are_folded():
    try:
        import cpp_engine
    except ImportError:
        try:
            from submission import cpp_engine
        except ImportError:
            pytest.skip("cpp_engine not built")
    toks = cpp_engine.tokenize("The U.S. F.B.I. and U.N. economy", remove_stop=True)
    assert "us" in toks
    assert "fbi" in toks
    assert "un" in toks
    assert "u" not in toks and "s" not in toks
    # Lone letters and spaced initials still drop.
    assert cpp_engine.tokenize("a I U. S.", remove_stop=True) == []
