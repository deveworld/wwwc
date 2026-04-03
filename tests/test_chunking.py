from tristore_bma.chunking import chunk_document


def test_chunk_document_overlap() -> None:
    text = " ".join(f"tok{i}" for i in range(20))
    chunks = chunk_document(text, chunk_size=6, overlap=2)

    assert len(chunks) == 5
    assert chunks[0].tokens == ["tok0", "tok1", "tok2", "tok3", "tok4", "tok5"]
    assert chunks[1].tokens[:2] == ["tok4", "tok5"]
