# Raw objective input (verbatim)

Source: user message opening the session (2026-06-07), with the Goodfire paper
`/Users/johan/Downloads/goodfirepaper_compressed.pdf` attached.

---

We're going to start a new research session. This will be a very long one that will require
multiple runs on cinaps.

Our goal will be to explore a map between the behavioural space where probability for weekday
token is high and for other weekdays is low, with the corresponding activation space.

For now, we need a new set of prompts. We need a set of prompt that give a lot of different
points in the probability simplex where sum of probability for weekdays is high, and other
probability is under <10%.

For this, this first step will be to start afresh with the prompt. We'll need a good number to do
PCA (compare with past code to approximate amount). The goal would be to span the relevant
subspace of the simplex quite well, with prompts like: Name a day of the week. Name a day of the
week that is not Tuesday. Name a day of the week that starts with T (and the converse). What is
the day after Tuesday? First output token should be the day of the week.

Decide the number of prompts needed, generate the prompts, and evaluate if they do give a nice
amount of coverage of the probability simplex by sending the prompts on cinaps with the llama
model and evaluating results. Do flag if anything appears missing, unclear, or if this could be
improved somewhat. Paper that originated the repo is attached.

---

## Paper identification (resolved during planning)

The attached PDF is **"Manifold Steering Reveals the Shared Geometry of Neural Network
Representation and Behavior"** (Wurgaft, Rager, Kowal, …, Goodman, Fel, Geiger, Lubana; Goodfire /
Stanford). Repo branch: `github.com/goodfire-ai/causalab/tree/manifold_steering`.

Relevant framing (paper §2.1):
- Behavior space **Y = Δ^|Z|**: the probability simplex over the conceptual domain
  Z = {Mon…Sun}, plus an "other" class for off-concept mass. `p(x)` = the model's next-token
  distribution restricted to the weekday tokens + other. (Exactly the user's "behavioural space.")
- Activation space **A = R^n**: a residual-stream activation `h(x)`.
- The paper fits an activation manifold **M_h** and a behavior manifold **M_y** and studies the
  link **M_h ↔ M_y** via manifold vs. linear steering.
- The paper's running weekday example uses ONLY the arithmetic family: "What day is k days after
  z?", z∈{Mon…Sun}, k∈{1..7} → distributions that concentrate on the answer and spread to
  cyclically-adjacent days. This clusters near simplex vertices; the user's diverse prompt
  families are an extension designed to span the simplex interior far more thoroughly.
