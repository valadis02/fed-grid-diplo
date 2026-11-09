"""Κοινά σύνολα επιτιθέμενων για gen_exp50.py (testbed) και tools/offline_sim.py."""

# Nested σύνολα επιτιθέμενων ανά ποσοστό (10 ⊂ 20 ⊂ 30 ⊂ 40), ισορροπημένα στους 2 fogs.
# Σειρά = τυχαία μετάθεση (rng 2026) των 30%, και μετά 3/2 επιπλέον σπίτια για το 40%.
# Το 30% είναι ακριβώς το ATTACKERS_30 (συνέχεια με το Byzantine πείραμα 1).
ORDER_FOG1 = [17, 3, 10, 15, 12, 2, 5, 1, 7, 21]
ORDER_FOG2 = [39, 29, 40, 50, 30, 42, 33, 44, 27, 36]
PER_FOG = {10: (2, 3), 20: (5, 5), 30: (7, 8), 40: (10, 10)}   # (fog1, fog2) από 25 το καθένα


def attackers_for(frac):
    k1, k2 = PER_FOG[frac]
    return sorted(ORDER_FOG1[:k1] + ORDER_FOG2[:k2])

assert all(set(attackers_for(a)) <= set(attackers_for(b)) for a, b in [(10, 20), (20, 30), (30, 40)])

