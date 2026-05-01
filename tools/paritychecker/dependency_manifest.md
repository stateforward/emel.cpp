# Parity Dependency Manifest

`parity_dependency_manifest/v1` is a deterministic line-oriented manifest emitted by the maintained
paritychecker CLI:

```sh
build/paritychecker_zig/paritychecker --write-dependency-manifest \
  tools/paritychecker/dependency_manifest.txt
```

The CLI path is the production emission path. Internally it is backed by
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
`source`, `config`, `fixture`, `model`, `script`, or `snapshot`.

`scripts/quality_gates.sh` checks the maintained baseline at
`tools/paritychecker/dependency_manifest.txt` before allowing automatic parity skips:

```sh
build/paritychecker_zig/paritychecker \
  --check-dependency-manifest tools/paritychecker/dependency_manifest.txt
```

Fresh output reports `full_gate=0 reason=fresh`. Missing, stale, or uncertain manifest data is
never a skip signal; it requires the relevant full parity gate. Operators can force the uncertain
path explicitly:

```sh
EMEL_PARITY_DEPENDENCY_MANIFEST_UNCERTAIN=1 scripts/quality_gates.sh
```

When the quality gate sees `full_gate=1` from the manifest check, it runs
`scripts/paritychecker.sh` even if changed-file inference would otherwise skip parity.
