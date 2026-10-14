from app.similar import GameFeatures, SimilarityIndex


def g(i, title, desc, genres, franchise=None, dev=None):
    return GameFeatures(i, f"g{i}", title, desc, genres, franchise, dev)


GAMES = [
    g(1, "Gears of War: E-Day", "Brutal third person cover shooter against the Locust horde", ["Third Person Shooter"], "Gears of War", "The Coalition"),
    g(2, "Gears 5", "Cover shooter campaign, Kait fights the Swarm horde", ["Third Person Shooter"], "Gears of War", "The Coalition"),
    g(3, "Planet Zoo 2", "Build and manage a zoo, care for animals and guests", ["Tycoon"]),
    g(4, "Zoo Tycoon", "Manage your zoo park and keep animals happy", ["Tycoon"]),
    g(5, "Boltgun 2", "Retro boomer shooter in the Warhammer universe", ["FPS"]),
]


def test_same_franchise_and_genre_rank_first():
    idx = SimilarityIndex(GAMES)
    top = idx.similar(1)
    assert top[0].slug == "g2"
    assert any("серия: Gears of War" in r for r in top[0].reasons)


def test_unrelated_games_are_not_returned():
    idx = SimilarityIndex(GAMES)
    slugs = [s.slug for s in idx.similar(3)]
    assert slugs[0] == "g4"
    assert "g1" not in slugs and "g2" not in slugs


def test_unknown_game_and_self_exclusion():
    idx = SimilarityIndex(GAMES)
    assert idx.similar(999) == []
    assert all(s.id != 5 for s in idx.similar(5))
