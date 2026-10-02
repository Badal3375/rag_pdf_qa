from rag.chunker import chunk_pages


def test_chunks_respect_size_and_page():
    text = " ".join(f"Sentence number {i} is here." for i in range(200))
    chunks = chunk_pages([(3, text)], chunk_size=300, overlap=50)
    assert len(chunks) > 3
    assert all(c.page == 3 for c in chunks)
    assert all(len(c.text) <= 400 for c in chunks)


def test_overlap_validation():
    import pytest
    with pytest.raises(ValueError):
        chunk_pages([(1, "x")], chunk_size=100, overlap=100)
