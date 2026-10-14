import pytest

from app.metacritic import ParseError, ReviewSample, Review, parse_browse_page, parse_new_releases, parse_product, parse_reviews


def test_new_releases_takes_only_that_section(fixture_text):
    slugs = parse_new_releases(fixture_text("game_index.html"))
    assert len(slugs) == 20
    assert slugs[0] == "over-the-hill"
    assert "gears-of-war-e-day" in slugs
    assert len(set(slugs)) == len(slugs)
    # the first link of the next section (Upcoming Games) is in the fixture and must be left out
    assert slugs[-1] == "dynasty-warriors-3-complete-edition-remastered"


def test_new_releases_missing_section():
    with pytest.raises(ParseError):
        parse_new_releases("<html><body>no sections</body></html>")


def test_browse_page_reads_only_the_result_grid(fixture_text):
    slugs = parse_browse_page(fixture_text("browse_page.html"))
    assert len(slugs) == 24
    assert "header-noise" not in slugs
    assert slugs[0] == "niph"


def test_parse_product(fixture_json):
    d = parse_product(fixture_json("composer_gears.json"))
    assert d.title == "Gears of War: E-Day"
    assert d.developer == "The Coalition"
    assert d.publisher == "Xbox Game Studios"
    assert d.cover_url.endswith("/provider/7/2/7-1788246859.jpg")  # cardImage (box art), not the wide header
    assert d.video_url.startswith("https://cdn.jwplayer.com/players/")
    assert d.genres == ["Third Person Shooter"]
    assert d.franchise == "Gears of War"
    assert [(p.name, p.metascore, p.is_lead) for p in d.platforms] == [("Xbox Series X", 88, True), ("PC", 83, False)]
    assert d.description and "\r" not in d.description


def test_parse_reviews_and_fingerprint(fixture_json):
    critic = parse_reviews(fixture_json("reviews_crit.json"), "critic")
    users = parse_reviews(fixture_json("reviews_user.json"), "user")
    assert len(critic) == 3 and critic[0].author == "DayOne" and critic[0].score == 100
    assert users and all(r.key for r in users)
    a = ReviewSample("critic", 74, {}, critic)
    b = ReviewSample("critic", 74, {}, list(reversed(critic)))
    assert a.fingerprint() == b.fingerprint()  # order does not matter
    c = ReviewSample("critic", 75, {}, critic)
    assert a.fingerprint() != c.fingerprint()  # a new review changes the total
    assert ReviewSample("critic", 0, {}, []).fingerprint() is None
