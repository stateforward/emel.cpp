# Parity Dependency Manifest

`parity_dependency_manifest/v1` is a deterministic line-oriented manifest emitted by
`emel::paritychecker::dependency_manifest::render()` and
`emel::paritychecker::dependency_manifest::write()`.

The first line is the schema string. The second line is the conservative freshness rule:

```text
full_gate_on=missing,stale,uncertain
```

Every remaining line is one dependency record:

```text
record runner=<runner> mode=<mode> kind=<kind> path=<repo-relative-path> reason=<reason>
```

Records are source-controlled and grouped by parity mode in runner order. `kind` is one of
`source`, `config`, `fixture`, `model`, `script`, or `snapshot`. Missing, stale, or uncertain
manifest data is never a skip signal; it requires the relevant full parity gate.
