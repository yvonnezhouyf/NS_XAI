"""
LogiEx formula templates per query type for the paratransit domain.

Templates use placeholders:
  {v1} = assigned vehicle ID
  {v2} = closest vehicle ID
  {r}  = request ID / decision epoch

These are filled at runtime by evaluator.py before parsing.
Formula strings follow the LogiEx grammar (transit_logics.py).
"""

LOGIEX_TEMPLATES = {
    1: [  # Why is Vehicle {v1} chosen over Vehicle {v2} at request {r}?
        "r({v1})", "r({v2})",
        "N(0,{v1})", "N(0,{v2})",
        "eta({v1})", "eta({v2})",
        "C({v1})", "C({v2})",
        "O(0,{v1})", "O(0,{v2})",
        "sp({r},{v1})", "sp({r},{v2})",
        "vcv(C({v1}),O(0,{v1}))",
        "vcv(C({v2}),O(0,{v2}))",
        "viod(tp({r}),eta({v1}))",
        "viod(tp({r}),eta({v2}))",
        "pctd(tp({r}),eta({v1}))",
        "pctd(tp({r}),eta({v2}))",
        "Phi3(r({v1}),r({v2}))",
    ],

    2: [  # At epoch {r}, which vehicle offers the most reliable service?
        "r({v1})", "r({v2})",
        "N(0,{v1})", "N(0,{v2})",
        "eta({v1})", "eta({v2})",
        "sp({r},{v1})", "sp({r},{v2})",
        "viod(tp({r}),eta({v1}))",
        "viod(tp({r}),eta({v2}))",
        "pctd(tp({r}),eta({v1}))",
        "pctd(tp({r}),eta({v2}))",
        "Phi3(r({v1}),r({v2}))",
    ],

    3: [  # At epoch {r}, why avoid/start carpooling for request?
        "C({v1})",
        "O(0,{v1})",
        "vcv(C({v1}),O(0,{v1}))",
        "vcvq(C({v1}),O(0,{v1}))",
        "r({v1})",
    ],

    4: [  # At what epoch does the dispatcher stop being overly pessimistic?
        "r({v1})",
        "N(0,{v1})",
    ],

    5: [  # At epoch {r}, how confident is the dispatcher about assigning Vehicle {v1}?
        "r({v1})",
        "N(0,{v1})",
        "eta({v1})",
        "sp({r},{v1})",
        "viod(tp({r}),eta({v1}))",
        "pctd(tp({r}),eta({v1}))",
    ],

    6: [  # At epoch {r}, why is the closest vehicle not assigned?
        "r({v1})", "r({v2})",
        "N(0,{v1})", "N(0,{v2})",
        "eta({v1})", "eta({v2})",
        "C({v1})", "C({v2})",
        "O(0,{v1})", "O(0,{v2})",
        "sp({r},{v1})", "sp({r},{v2})",
        "vcv(C({v1}),O(0,{v1}))",
        "vcv(C({v2}),O(0,{v2}))",
        "viod(tp({r}),eta({v1}))",
        "viod(tp({r}),eta({v2}))",
        "pctd(tp({r}),eta({v1}))",
        "pctd(tp({r}),eta({v2}))",
        "Phi3(r({v1}),r({v2}))",
    ],

    7: [  # What is the scheduled pickup/dropoff time for request {r}?
        "tp({r})",
        "td({r})",
        "eta({v1})",
    ],

    8: [  # What is the passenger count/capacity pressure for Vehicle {v1}?
        "C({v1})",
        "O(0,{v1})",
        "vcvq(C({v1}),O(0,{v1}))",
    ],

    9: [  # How does Vehicle {v1} compare to Vehicle {v2} on delay risk?
        "r({v1})", "r({v2})",
        "N(0,{v1})", "N(0,{v2})",
        "eta({v1})", "eta({v2})",
        "C({v1})", "C({v2})",
        "O(0,{v1})", "O(0,{v2})",
        "sp({r},{v1})", "sp({r},{v2})",
        "viod(tp({r}),eta({v1}))",
        "viod(tp({r}),eta({v2}))",
        "pctd(tp({r}),eta({v1}))",
        "pctd(tp({r}),eta({v2}))",
        "Phi3(r({v1}),r({v2}))",
    ],

    10: [  # At request {r}, how many vehicles are available right now?
        "availablecar({r})",
    ],

    11: [  # Case 1: why is a farther vehicle chosen at request {r}?
        "r({v1})", "r({v2})",
        "N(0,{v1})", "N(0,{v2})",
        "eta({v1})", "eta({v2})",
        "sp({r},{v1})", "sp({r},{v2})",
        "C({v1})", "C({v2})",
        "O(0,{v1})", "O(0,{v2})",
        "vcv(C({v1}),O(0,{v1}))",
        "vcv(C({v2}),O(0,{v2}))",
        "viod(tp({r}),eta({v1}))",
        "viod(tp({r}),eta({v2}))",
        "pctd(tp({r}),eta({v1}))",
        "pctd(tp({r}),eta({v2}))",
        "Phi3(r({v1}),r({v2}))",
    ],

    12: [  # Case 2: how did the event affect assignment at epoch {r}?
        "r({v1})", "r({v2})",
        "N(0,{v1})", "N(0,{v2})",
        "eta({v1})", "eta({v2})",
        "sp({r},{v1})", "sp({r},{v2})",
        "viod(tp({r}),eta({v1}))",
        "viod(tp({r}),eta({v2}))",
        "pctd(tp({r}),eta({v1}))",
        "pctd(tp({r}),eta({v2}))",
        "Phi3(r({v1}),r({v2}))",
    ],

    14: [  # How do traffic and route burden affect timing?
        "eta({v1})", "eta({v2})",
        "sp({r},{v1})", "sp({r},{v2})",
        "r({v1})", "r({v2})",
        "Phi3(r({v1}),r({v2}))",
    ],

    15: [  # At event epoch, how much does congestion affect timing?
        "eta({v1})",
        "r({v1})",
        "viod(tp({r}),eta({v1}))",
        "pctd(tp({r}),eta({v1}))",
        "sp({r},{v1})",
    ],

    16: [  # What is the current traffic level? (no LogiEx formula for this)
        "tp({r})",
        "td({r})",
    ],

    17: [  # How many pending requests on the assigned vehicle?
        "sp({r},{v1})",
    ],

    18: [  # How many pending requests on the closest vehicle?
        "sp({r},{v2})",
    ],

    19: [  # What is the assigned vehicle ETA to pickup?
        "eta({v1})",
    ],

    20: [  # What is the assigned vehicle ETA to dropoff?
        "eta({v1})",
        "td({r})",
    ],

    21: [  # What is the closest vehicle ETA to pickup?
        "eta({v2})",
    ],

    22: [  # What is the closest vehicle ETA to dropoff?
        "eta({v2})",
        "td({r})",
    ],

    23: [  # How often does the model update?
        "N(0,{v1})",
    ],

    24: [  # How does the event affect the traffic?
        "tp({r})",
        "td({r})",
    ],

    25: [  # How do assigned and closest differ in workload?
        "sp({r},{v1})", "sp({r},{v2})",
        "O(0,{v1})", "O(0,{v2})",
        "Phi4(sp({r},{v1}),sp({r},{v2}))",
    ],

    26: [  # How much later is assigned vs closest ETA?
        "eta({v1})", "eta({v2})",
        "sp({r},{v1})", "sp({r},{v2})",
    ],

    27: [  # What is the assigned vehicle's expected ride time?
        "eta({v1})",
        "tp({r})",
        "td({r})",
        "sp({r},{v1})",
    ],

    28: [  # What is the closest vehicle's expected ride time?
        "eta({v2})",
        "tp({r})",
        "td({r})",
        "sp({r},{v2})",
    ],

    # Fallback: comprehensive set of formulas
    -1: [
        "tp({r})", "td({r})",
        "r({v1})", "r({v2})",
        "N(0,{v1})", "N(0,{v2})",
        "eta({v1})", "eta({v2})",
        "C({v1})", "C({v2})",
        "O(0,{v1})", "O(0,{v2})",
        "sp({r},{v1})", "sp({r},{v2})",
        "vcv(C({v1}),O(0,{v1}))",
        "vcv(C({v2}),O(0,{v2}))",
        "viod(tp({r}),eta({v1}))",
        "viod(tp({r}),eta({v2}))",
        "pctd(tp({r}),eta({v1}))",
        "pctd(tp({r}),eta({v2}))",
        "Phi3(r({v1}),r({v2}))",
        "availablecar({r})",
        "car({r})",
    ],
}
