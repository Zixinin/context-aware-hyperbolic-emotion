"""CPU-only calibration from sealed REAL fresh-model gradient Gram statistics.

No torch, model construction, optimizer, data selection or metric search.
"""
import math
import statistics

RULE = {
    'version': 'fresh_train_log_median_shared_encoder_v1',
    'reference_gamma': 0.1,
    'selection': 'all ten previously fixed train-only cohorts, equal cohort weight',
    'formula': 'gamma_G = exp(median_b(log(0.1 * norm(g_old_b) / norm(g_G_b))))',
    'stored_norms': 'both saved auxiliary gradients already contain reference gamma=0.1',
    'application': 'one constant multiplying the ENTIRE mean(soft-target CE + positive distance)',
    'direction': 'report ALL BCE/aux and old/new auxiliary cosines; never select gamma by cosine',
    'no_search': True,
}


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def number(x):
    value = float(x)
    need(math.isfinite(value), 'Nonfinite gradient statistic')
    return value


def cosine(x):
    value = number(x)
    need(-1.0000001 <= value <= 1.0000001, 'Invalid cosine')
    return max(-1., min(1., value))


def sum_norm(a, b, cos):
    """Exact norm from the two gradients' saved Gram entries, to FP64 rounding."""
    a, b = number(a), number(b)
    need(a >= 0 and b >= 0, 'Negative gradient norm')
    if a == 0 or b == 0:
        need(cos in [None, ''] or abs(number(cos)) <= 1.0000001, 'Invalid zero-vector cosine')
        return math.hypot(a, b)
    c = cosine(cos)
    # This form avoids subtraction of near-equal large terms at cosine=-1.
    return math.sqrt((a-b)**2 + 2*a*b*(1+c))


def calibrate(shared_rows):
    need(len(shared_rows) == 10, 'Expected ALL ten fixed fresh cohorts')
    need(len({r['cohort'] for r in shared_rows}) == 10, 'Duplicate cohort')
    logs = []
    for r in shared_rows:
        need(r['module'] == 'shared_encoder' and r['tag'] == 'fresh_initial', 'Calibration must use fresh encoder only')
        old, new = number(r['coordinate_aux_norm']), number(r['geodesic_aux_norm'])
        need(old > 0 and new > 0 and number(r['bce_norm']) > 0, 'Zero auxiliary/BCE norm: stop, no epsilon fallback')
        # Saved norms = 0.1 times the respective UNWEIGHTED auxiliary gradients.
        logs.append(math.log(.1) + math.log(old) - math.log(new))
    gamma = math.exp(statistics.median(logs))
    need(math.isfinite(gamma) and gamma > 0, 'Invalid calibrated coefficient')
    rows = []
    for r in shared_rows:
        a, old, new = (number(r[k]) for k in ['bce_norm', 'coordinate_aux_norm', 'geodesic_aux_norm'])
        co, cn = cosine(r['bce_coordinate_aux_cosine']), cosine(r['bce_geodesic_aux_cosine'])
        matched = new * (gamma/.1)
        ratio = matched/old
        rows.append(dict(cohort=r['cohort'], individual_exact_matching_gamma=.1*old/new,
            bce_norm=a, old_weighted_aux_norm=old, geodesic_weighted_aux_norm_at_point1=new,
            geodesic_weighted_aux_norm_at_gamma_G=matched, matched_to_old_aux_norm=ratio,
            old_aux_to_bce=old/a, matched_aux_to_bce=matched/a,
            bce_old_aux_cosine=co, bce_geodesic_aux_cosine=cn,
            old_new_aux_cosine=cosine(r['old_new_aux_cosine']),
            geodesic_direction_conflicts_with_bce=cn < 0,
            old_shared_total_norm=sum_norm(a,old,co),
            matched_shared_total_norm=sum_norm(a,matched,cn)))
    log_residuals = [math.log(r['matched_to_old_aux_norm']) for r in rows]
    need(abs(statistics.median(log_residuals)) < 1e-12, 'Log-median matching failed')
    return gamma, rows


def module_contributions(rows, gamma):
    """Rescale the new aux Gram matrix, leaving BCE fixed for every module."""
    need(number(gamma) > 0, 'Coefficient must be positive')
    result = []
    for r in rows:
        a, old, new = (number(r[k]) for k in ['bce_norm','coordinate_aux_norm','geodesic_aux_norm'])
        co = None if r['bce_coordinate_aux_cosine'] in ['', None] else cosine(r['bce_coordinate_aux_cosine'])
        cn = None if r['bce_geodesic_aux_cosine'] in ['', None] else cosine(r['bce_geodesic_aux_cosine'])
        old_total, original_new = sum_norm(a,old,co), sum_norm(a,new,cn)
        need(math.isclose(old_total,number(r['coordinate_total_norm']),rel_tol=2e-6,abs_tol=1e-9), 'Saved old Gram entries inconsistent')
        need(math.isclose(original_new,number(r['geodesic_total_norm']),rel_tol=2e-6,abs_tol=1e-9), 'Saved new Gram entries inconsistent')
        matched = new * gamma/.1
        result.append(dict(cohort=r['cohort'], module=r['module'], bce_norm=a,
            old_weighted_aux_norm=old, matched_weighted_aux_norm=matched,
            bce_geodesic_aux_cosine=cn, old_total_norm=old_total,
            matched_total_norm=sum_norm(a,matched,cn)))
    return result
