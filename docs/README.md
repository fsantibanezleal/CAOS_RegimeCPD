# regimecpd wiki

The theory, the equations, the primary sources and the honest limits of everything this package does.

This wiki is written as the package is versioned, not afterwards. A page appears in the same release as
the code it documents, so if a method is listed here it exists, and if it is not listed here it does not.

## Themes

| Theme | What it covers |
|---|---|
| [Architecture](architecture.md) | The data contract, and the machinery that keeps the two arms of the comparison honest |
| [Methods](methods.md) | One deep page per rung of the detector ladder: theory, equations, references, limits |
| [Guides](guides.md) | Install, run it on your own data, and how to read what comes back |

## What this package IS

A library for detecting the onset of abnormal condition on machines whose **operating context varies**,
and for measuring whether conditioning on that context is worth doing. It provides the segmentation, the
residual, a ladder of detectors from Shewhart to conformal-calibrated change-point methods, and a
measurement layer built specifically so that two methods can be compared at the same false-alarm budget.

## What this package is NOT

- **Not a truck library.** It has no notion of a truck, a tyre or a strut. The domain layer belongs to
  whatever consumes it. The reference validation runs on turbofan data precisely because a
  regime-conditional detector that only works on trucks would not deserve to be a package.
- **Not a forecasting library.** It answers "has something changed, when, and which channel is
  implicated", not "what will the value be next week".
- **Not a root-cause tool.** Attribution here narrows a candidate set. Contribution-style attribution
  smears across correlated channels, and this package says so wherever it returns one. See
  [Attribution and its limits](architecture/01_data-contract.md#attribution-and-why-it-is-not-a-cause).
- **Not a claim that regime conditioning always helps.** It is instrumentation for finding out. A null
  result is a supported and reportable outcome.

## The one idea

Every channel on a load-varying machine moves with the load. A change detector on raw channels detects
the load. So:

```
raw channels -> regime segmentation -> within-regime residual -> change point -> onset
```

and the question worth asking is not "does this detect things" but "does the regime stage reduce false
alarms at a fixed detection delay, and by how much, with what interval". Everything in this package is
shaped to answer that question rather than to assume its answer.
