from rag_agents.web.rendering import render_answer


def test_valid_citations_become_links_invalid_stay_text() -> None:
    html = render_answer("Смысл в вере [1], см. также [7].", "src-m", citation_count=2)
    assert '<a href="#src-m-1" class="cite" data-cite="1">[1]</a>' in html
    assert "[7]" in html
    assert "src-m-7" not in html


def test_html_and_scripts_are_not_rendered() -> None:
    html = render_answer('<script>alert(1)</script> <img src=x onerror="x()"> **жирный**', "a", 0)
    assert "<script" not in html
    assert "<img" not in html
    assert "<strong>жирный</strong>" in html
