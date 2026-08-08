# 06 mSTAMP: the multidimensional matrix profile

Yeh, Kavantzas and Keogh (2017). The **second** answer in this package to "which variables", reached by a
route that shares nothing with the PCA contributions.

That independence is the point. Agreement between two unrelated attributions is worth considerably more
than either alone, and disagreement is a signal to look harder rather than to pick a favourite.

## What a matrix profile is

For every subsequence of length $m$, record the z-normalised Euclidean distance to its nearest neighbour
elsewhere in the record. Small means "this pattern happens again" (a motif). **Large** means "nothing else
here looks like this", which is a discord, and discords are what onset detection wants.

No labels, no training set, no threshold tuned before the fact.

The distances come from MASS: the sliding dot product is an FFT convolution, so a distance profile is
$O(n\log n)$ rather than $O(nm)$. Without that the full profile is $O(n^2m)$ and unusable past a few
hundred samples. `test_matches_a_directly_computed_znormalised_distance` checks MASS against the
definition written out the slow way, because an off-by-one in the correlation lag produces distances that
are entirely plausible and systematically wrong.

## The multidimensional part

The obvious extension to $d$ channels is to sum across all of them. The authors show this does not produce
meaningful motifs except in contrived cases: a pattern repeating in three of twelve channels is swamped by
the nine carrying noise.

mSTAMP instead uses the **best** $k$ dimensions. Sorting the per-dimension distances ascending and taking
a running mean gives every $k$ at once:

$$D_k(i,j) = \frac{1}{k}\sum_{l=1}^{k}\mathrm{sort}_l\left(d_1(i,j),\dots,d_d(i,j)\right), \qquad
P_k(i) = \min_j D_k(i,j)$$

$P_k$ is non-decreasing in $k$ by construction, and there is a test asserting it: a violation means the
sort or the cumulative mean is wrong.

Choosing $k$ is deliberately left to the user. A fault in two channels and a fault in nine are different
events. The authors give a minimum-description-length heuristic; it is not implemented here, and its
absence is stated rather than papered over.

## THE finding: the matching subset names the wrong channels at a discord

mSTAMP's $k$-dimensional subset is the $k$ channels with the **smallest** distances, because the algorithm
was designed to find motifs. At a discord that names the channels which still look **normal**.

Measured on a four-channel record with the discord planted in `b` and `d`:

| method | returns |
|---|---|
| `matching_subset(k=2)` | `{a, c}` the two channels still repeating cleanly |
| `anomalous_channels` | `{b, d}` the faulted ones |

Reading the matching subset as attribution reports the two **healthy** channels as the culprits, with a
perfectly sensible-looking discord peak beside it. Both halves are asserted by
`test_the_MATCHING_subset_names_exactly_the_wrong_channels_at_a_discord`, so the distinction stays
load-bearing rather than becoming a comment somebody deletes.

So the API has two separate methods and they answer different questions:

- **`matching_subset`** is mSTAMP's own quantity: in which $k$ channels does this window resemble
  something else. A motif question.
- **`anomalous_channels`** is the discord attribution: per-channel distance to the full-dimensional
  nearest neighbour, large meaning that channel refuses to match.

### The same asymmetry decides which k detects a partial fault

$P_k$ uses the $k$ smallest distances, so a fault confined to two of four channels is **invisible at
$k=2$** (the two clean channels still match) and appears at $k=d$, where the failing channels can no
longer be excluded. `test_a_partial_fault_is_invisible_at_low_k_and_shows_at_high_k` pins it.

**For fault detection, prefer a high $k$.** This is the opposite of the intuition that a fault in few
channels should be found at low $k$.

## A measured correction to the exclusion-zone folklore

The standard telling: without a trivial-match exclusion zone, every subsequence matches the one beside it,
the profile collapses to zero, and the method reports an ordinary record no matter what is in it.

That is the claim a test was originally written to assert here, and **it failed**. Measured on a
stochastic smooth signal, window 40, median profile:

| smoothing | lag-1 correlation | no zone | zone 0.5 | ratio |
|---|---|---|---|---|
| 30 | 0.959 | 0.291 | 0.322 | 0.91 |
| 100 | 0.987 | 0.323 | 0.374 | 0.87 |
| 300 | 0.998 | 0.296 | 0.334 | 0.89 |

About a tenth, not a collapse, and it saturates by a zone of roughly four samples. z-normalisation is the
reason: it removes the mean and the scale, so what remains is pure shape, and a window's shape changes
measurably from one sample to the next even when the raw signal barely moves.

Three framings were tried before this one and all would have passed for the wrong reason: a noise discord
survives without an exclusion zone (noise does not match its own shifted copy either), a high-frequency
discord survives for the same reason, and a single planted ramp cannot show it because any shift of a
short anomaly pulls in samples from outside it.

**Practical consequence, worth carrying into the product: for this class of data the exclusion zone is not
the critical parameter, the window length is.** The 0.5 default is kept because it costs nothing and does
matter for near-exactly-repeating motifs, which is where the folklore comes from.

A related fix: `exclusion=0` now means exactly "only the self-match is removed". It was floored at 1,
which silently applied a small zone even when none was asked for and made the parameter impossible to
test at its own boundary.

## Other design notes

**Normalisation across channels.** Each dimension's distance is divided by the maximum possible for the
window before the sort. Without it, a channel that happens to be noisier contributes larger distances at
every $k$ and dominates subset selection through its variance rather than through carrying the pattern.

**The statistic sits at the window START.** Placing it at the centre or the end shifts every reported
onset by a fixed amount, which is invisible in a plot and fatal in a delay metric. The trailing $m-1$
samples are undefined because no window begins there.

**No sign flip.** A large profile already means more anomalous. Worth stating because most other rungs
here do need one.

**A gapped series is rejected**, not filled. z-normalisation is undefined over a gap, and quietly
interpolating would invent the very shape the method is about to measure.

**Flat windows.** A window with zero variance has no z-normalised form, so its distance is set to the
maximum for the window length: a flat window matches nothing because it has no shape to match. A flat
query against a flat window is a genuine match at distance zero, and both cases are tested.

## Reference

Yeh, C.-C. M., Kavantzas, N., Keogh, E. J. "Matrix Profile VI: Meaningful Multidimensional Motif
Discovery." *ICDM 2017*, pp. 565-574. Reference implementation: <https://github.com/mcyeh/mstamp>
