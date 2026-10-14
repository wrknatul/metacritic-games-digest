from app.letsplay import VideoCandidate, parse_vtt, pick_letsplay, title_matches

# real search results for "Gears of War E-Day let's play" (2026-10-14)
GEARS = [
    VideoCandidate("90Fs3mw6mcQ", "GEARS OF WAR E-DAY Walkthrough Gameplay Part 1 - INTRO (FULL GAME)", "theRadBrad", 867376, 15087),
    VideoCandidate("HuDT7sRalc0", "Gears of War E-Day - Part 1 - MARCUS & DOM ARE BACK", "GoldGlove Let's Plays", 36381, 3655),
    VideoCandidate("rsTU_2mJwCY", "Gears of War E-Day - Part 1 - WHERE IT ALL BEGAN..", "Hollow", 1019030, 21920),
    VideoCandidate("_PBCK1jgkHQ", "Gears of War E-Day is UNBELIEVABLY Great", "Kevduit", 342536, 1959),
    VideoCandidate("x1", "Gears of War E-Day - Official Launch Trailer", "Xbox", 5000000, 150),
    VideoCandidate("x2", "Gears of War E-Day: 15 Things You Need To Know Before You Buy", "GamingBolt", 2000000, 900),
    VideoCandidate("x3", "Gears of War 3 Full Playthrough", "Someone", 9000000, 30000),
]


def test_picks_most_viewed_real_letsplay():
    best = pick_letsplay("Gears of War: E-Day", GEARS)
    assert best.id == "rsTU_2mJwCY"


def test_rejects_other_games_trailers_and_guides():
    assert not title_matches("Gears of War: E-Day", "Gears of War 3 Full Playthrough")
    only_noise = [c for c in GEARS if c.id in ("x1", "x2", "x3")]
    assert pick_letsplay("Gears of War: E-Day", only_noise) is None


def test_short_title_needs_every_word():
    assert title_matches("Niph", "NIPH - Let's Play Part 1")
    assert not title_matches("Planet Zoo 2", "Planet Zoo Let's Play")


VTT = """WEBVTT
Kind: captions
Language: en

00:00:00.000 --> 00:00:02.500 align:start position:0%

hello<00:00:00.500><c> everyone</c>

00:00:02.500 --> 00:00:02.510 align:start position:0%
hello everyone


00:00:02.510 --> 00:00:05.000 align:start position:0%
hello everyone
welcome<00:00:03.000><c> back</c>

00:00:05.000 --> 00:00:07.000 align:start position:0%
welcome back
to the channel

00:10:00.000 --> 00:10:02.000 align:start position:0%
this is too late
"""


def test_parse_vtt_removes_rolling_repeats_and_cuts_by_time():
    assert parse_vtt(VTT, max_seconds=120) == "hello everyone welcome back to the channel"
