"""Prospective eligible-data gate. No provider calls, ranking reconstruction or training."""
def require_ready(status):
    expected = {"valid_api_rankings":6036, "missing_rankings":4, "eligible_missing_rankings":0,
                "preserved_failed_rankings":2, "unattempted_paired_exclusions":2}
    if any(type(status.get(k)) is not int or status[k] != v for k,v in expected.items()):
        raise ValueError("eligible API dataset is not complete and symmetric")
    return {"eligible_complete":True, "original_full_valid_complete":False,
            "eligible_candidate_sets":3018, "excluded_candidate_sets":2}
