"""Hand-curated F1 track physical properties for the 26 races in this dataset.

Driver/Year info in the dataset is synthetic, but the Race names map to real F1
circuits. Per-track properties (pit lane time loss, tyre abrasion, corner count,
typical pit-stops-per-race, etc.) are stable physical facts of the circuits and
should give the model a signal nobody is using in public notebooks (I checked).

The numbers below are drawn from public F1 reference data (Pirelli wear ratings,
team radio analyses, official FIA pit lane time loss tables). Sources mostly:
official F1.com circuit stats, Pirelli tyre selection, Mercedes/RB strategy
briefings, and Race Fans / The Race articles.

If a Race is unknown the row gets the dataset median, so the model degrades
gracefully on any future race name.
"""
from __future__ import annotations

import pandas as pd


TRACK_PROPS: dict[str, dict] = {
    # name → {length_km, n_corners, pit_loss_s, abrasion_1to5, downforce_1to5,
    #         drs_zones, typical_stops, street_track, sc_prob_pct, elev_change_m}
    "Bahrain Grand Prix":            {"length_km":5.412,"n_corners":15,"pit_loss_s":22.5,"abrasion":4,"downforce":3,"drs_zones":3,"typical_stops":2.0,"street":0,"sc_prob_pct":18,"elev_change_m":17},
    "Saudi Arabian Grand Prix":      {"length_km":6.174,"n_corners":27,"pit_loss_s":18.5,"abrasion":2,"downforce":3,"drs_zones":3,"typical_stops":1.0,"street":1,"sc_prob_pct":50,"elev_change_m":11},
    "Australian Grand Prix":         {"length_km":5.278,"n_corners":14,"pit_loss_s":19.0,"abrasion":3,"downforce":4,"drs_zones":4,"typical_stops":1.5,"street":1,"sc_prob_pct":40,"elev_change_m":35},
    "Emilia Romagna Grand Prix":     {"length_km":4.909,"n_corners":19,"pit_loss_s":24.0,"abrasion":3,"downforce":4,"drs_zones":1,"typical_stops":1.0,"street":0,"sc_prob_pct":33,"elev_change_m":32},
    "Miami Grand Prix":              {"length_km":5.412,"n_corners":19,"pit_loss_s":20.0,"abrasion":3,"downforce":3,"drs_zones":3,"typical_stops":1.5,"street":1,"sc_prob_pct":40,"elev_change_m":5},
    "Monaco Grand Prix":             {"length_km":3.337,"n_corners":19,"pit_loss_s":23.0,"abrasion":1,"downforce":5,"drs_zones":1,"typical_stops":1.0,"street":1,"sc_prob_pct":55,"elev_change_m":42},
    "Spanish Grand Prix":            {"length_km":4.657,"n_corners":14,"pit_loss_s":22.0,"abrasion":4,"downforce":4,"drs_zones":2,"typical_stops":2.0,"street":0,"sc_prob_pct":10,"elev_change_m":30},
    "Canadian Grand Prix":           {"length_km":4.361,"n_corners":14,"pit_loss_s":17.5,"abrasion":3,"downforce":3,"drs_zones":3,"typical_stops":1.5,"street":1,"sc_prob_pct":55,"elev_change_m":10},
    "Austrian Grand Prix":           {"length_km":4.318,"n_corners":10,"pit_loss_s":20.5,"abrasion":4,"downforce":3,"drs_zones":3,"typical_stops":2.0,"street":0,"sc_prob_pct":20,"elev_change_m":65},
    "British Grand Prix":            {"length_km":5.891,"n_corners":18,"pit_loss_s":19.5,"abrasion":4,"downforce":4,"drs_zones":2,"typical_stops":2.0,"street":0,"sc_prob_pct":25,"elev_change_m":24},
    "Hungarian Grand Prix":          {"length_km":4.381,"n_corners":14,"pit_loss_s":19.5,"abrasion":2,"downforce":5,"drs_zones":2,"typical_stops":1.5,"street":0,"sc_prob_pct":15,"elev_change_m":36},
    "Belgian Grand Prix":            {"length_km":7.004,"n_corners":19,"pit_loss_s":16.0,"abrasion":3,"downforce":2,"drs_zones":2,"typical_stops":1.5,"street":0,"sc_prob_pct":35,"elev_change_m":102},
    "Dutch Grand Prix":              {"length_km":4.259,"n_corners":14,"pit_loss_s":19.0,"abrasion":3,"downforce":4,"drs_zones":2,"typical_stops":1.5,"street":0,"sc_prob_pct":30,"elev_change_m":21},
    "Italian Grand Prix":            {"length_km":5.793,"n_corners":11,"pit_loss_s":17.0,"abrasion":3,"downforce":1,"drs_zones":2,"typical_stops":1.0,"street":0,"sc_prob_pct":20,"elev_change_m":18},
    "Azerbaijan Grand Prix":         {"length_km":6.003,"n_corners":20,"pit_loss_s":19.5,"abrasion":2,"downforce":3,"drs_zones":2,"typical_stops":1.0,"street":1,"sc_prob_pct":60,"elev_change_m":11},
    "Singapore Grand Prix":          {"length_km":4.940,"n_corners":19,"pit_loss_s":24.5,"abrasion":2,"downforce":5,"drs_zones":3,"typical_stops":1.5,"street":1,"sc_prob_pct":75,"elev_change_m":9},
    "Japanese Grand Prix":           {"length_km":5.807,"n_corners":18,"pit_loss_s":21.0,"abrasion":4,"downforce":4,"drs_zones":2,"typical_stops":2.0,"street":0,"sc_prob_pct":30,"elev_change_m":40},
    "Qatar Grand Prix":              {"length_km":5.380,"n_corners":16,"pit_loss_s":22.0,"abrasion":5,"downforce":3,"drs_zones":2,"typical_stops":2.5,"street":0,"sc_prob_pct":15,"elev_change_m":13},
    "United States Grand Prix":      {"length_km":5.513,"n_corners":20,"pit_loss_s":20.5,"abrasion":3,"downforce":4,"drs_zones":2,"typical_stops":1.5,"street":0,"sc_prob_pct":25,"elev_change_m":41},
    "Mexico City Grand Prix":        {"length_km":4.304,"n_corners":17,"pit_loss_s":20.0,"abrasion":2,"downforce":5,"drs_zones":3,"typical_stops":1.0,"street":0,"sc_prob_pct":40,"elev_change_m":13},
    "São Paulo Grand Prix":          {"length_km":4.309,"n_corners":15,"pit_loss_s":17.0,"abrasion":4,"downforce":3,"drs_zones":2,"typical_stops":2.0,"street":0,"sc_prob_pct":50,"elev_change_m":40},
    "Las Vegas Grand Prix":          {"length_km":6.201,"n_corners":17,"pit_loss_s":18.5,"abrasion":2,"downforce":2,"drs_zones":2,"typical_stops":1.0,"street":1,"sc_prob_pct":45,"elev_change_m":6},
    "Abu Dhabi Grand Prix":          {"length_km":5.281,"n_corners":16,"pit_loss_s":18.5,"abrasion":3,"downforce":4,"drs_zones":2,"typical_stops":1.0,"street":0,"sc_prob_pct":20,"elev_change_m":14},
    "Chinese Grand Prix":            {"length_km":5.451,"n_corners":16,"pit_loss_s":18.5,"abrasion":4,"downforce":4,"drs_zones":2,"typical_stops":2.0,"street":0,"sc_prob_pct":30,"elev_change_m":14},
    "French Grand Prix":             {"length_km":5.842,"n_corners":15,"pit_loss_s":20.5,"abrasion":3,"downforce":3,"drs_zones":2,"typical_stops":1.5,"street":0,"sc_prob_pct":15,"elev_change_m":18},
    # Pre-Season Testing is held at Bahrain — same physical track.
    "Pre-Season Testing":            {"length_km":5.412,"n_corners":15,"pit_loss_s":22.5,"abrasion":4,"downforce":3,"drs_zones":3,"typical_stops":2.0,"street":0,"sc_prob_pct":0,"elev_change_m":17},
}


TRACK_COLS = ["length_km", "n_corners", "pit_loss_s", "abrasion", "downforce",
              "drs_zones", "typical_stops", "street", "sc_prob_pct", "elev_change_m"]


def attach_track_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add `trk_*` columns by joining Race name to TRACK_PROPS. Unknown races get medians."""
    props = pd.DataFrame.from_dict(TRACK_PROPS, orient="index")
    medians = props[TRACK_COLS].median().to_dict()
    out = df.copy()
    for col in TRACK_COLS:
        out[f"trk_{col}"] = out["Race"].astype(str).map(lambda r: TRACK_PROPS.get(r, medians)[col]).astype(float)
    # Some useful interactions
    out["trk_pit_per_lap"] = out["trk_typical_stops"] / out["trk_length_km"].clip(lower=1.0)
    out["trk_corners_per_km"] = out["trk_n_corners"] / out["trk_length_km"].clip(lower=1.0)
    out["trk_downforce_x_corners"] = out["trk_downforce"] * out["trk_n_corners"]
    return out
