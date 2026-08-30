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
