def calculate_drift_score(adwin_flags, psi_scores, mape_score):

    score = 0

    if any(adwin_flags.values()):
        score += 2

    if any(v > 0.25 for v in psi_scores.values()):
        score += 1

    if mape_score > 20:
        score += 1

    return score


def select_model(score):

    if score == 0:
        return "VECM"

    elif score == 1:
        return "VECM_RETRAIN"

    else:
        return "PROPHET"
