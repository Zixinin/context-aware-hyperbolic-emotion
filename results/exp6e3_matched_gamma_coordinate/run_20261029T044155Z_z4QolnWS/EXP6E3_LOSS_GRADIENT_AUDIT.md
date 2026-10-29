# Original coordinate gradient parity

Full-precision original A1 fresh train-only statistics replayed over40 module/cohort rows.
Unweighted coordinate auxiliary/BCE/q are unchanged; weighted aux vector equals old .1 aux vector*(gamma_G/.1).
Every connected module finite; classifier receives BCE only; HNN/decoder receive aux only; shared encoder receives both.
Module cosine/norm and CE-negative-dot decomposition: probes/fresh_initial/MODULE_GRADIENTS.csv.
No detach or geodesic objective in the training forward; gamma was not recalibrated.
