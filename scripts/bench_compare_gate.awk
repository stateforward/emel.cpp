# bench_compare_gate.awk
#
# Compare-gate core for scripts/bench.sh. Reads the committed baseline
# (snapshots/bench/benchmarks.txt) as the first file and the current-run
# snapshot as the second file, then enforces regression tolerances and
# baseline coverage.
#
# Extracted from the inline awk in scripts/bench.sh so the gate logic has a
# single home and can be exercised directly by tools/bench/quality_gates_tests.
#
# Variables (all passed with -v):
#   tol                relative regression tolerance (e.g. 0.30)
#   abs_tol            absolute regression tolerance in ns (e.g. 5000)
#   strict_regression  1 => regressions fail; 0 => regressions warn only
#   scoped             1 => suite-scoped run (relaxes full-coverage checks)
#   host_arch          normalized host architecture (e.g. x86_64, aarch64).
#                      Baseline rows that name a *different* known bench
#                      architecture as a path segment are expected-absent on
#                      this host because the runner skips them via
#                      case_supported_on_host (tools/bench/bench_runner.cpp).
#                      Those rows are not treated as missing. Rows that name the
#                      host arch, or no known arch, stay required.
#
# A known bench architecture appears as a "/<arch>/" path segment in a case
# name (for example kernel/x86_64/op_add or flash_attention/aarch64/...).

function register_known_arch(a) {
  known_arch[a] = 1;
}

# Return the known bench architecture named as a path segment in `name`, or ""
# when the name does not embed a known architecture.
function named_arch(name,    a) {
  for (a in known_arch) {
    if (index(name, "/" a "/") > 0) {
      return a;
    }
  }
  return "";
}

function parse_entry(line, dest,    n, fields, name, ns, i, pair) {
  n = split(line, fields, " ");
  name = fields[1];
  for (i = 2; i <= n; ++i) {
    if (fields[i] ~ /^ns_per_op=/) {
      split(fields[i], pair, "=");
      ns = pair[2];
      break;
    }
  }
  if (name == "" || ns == "") {
    return;
  }
  dest[name] = ns;
}

BEGIN {
  # Architectures the bench runner gates on the host at build/run time.
  register_known_arch("x86_64");
  register_known_arch("aarch64");
}

FNR == NR {
  if ($0 ~ /^#/) {
    skip_base = ($0 ~ /proof_status=measurement_only/);
    next;
  }
  if (skip_base) {
    skip_base = 0;
    next;
  }
  parse_entry($0, base);
  next;
}
{
  if ($0 ~ /^#/) {
    skip_curr = ($0 ~ /proof_status=measurement_only/);
    next;
  }
  if (skip_curr) {
    skip_curr = 0;
    next;
  }
  parse_entry($0, curr);
  next;
}
END {
  fail = 0;
  compared = 0;
  for (name in curr) {
    if (!(name in base)) {
      print "error: new benchmark entry without baseline: " name > "/dev/stderr";
      fail = 1;
      continue;
    }
    compared += 1;
    relative_limit = base[name] * (1 + tol);
    absolute_limit = base[name] + abs_tol;
    if (curr[name] > relative_limit && curr[name] > absolute_limit) {
      limit = relative_limit > absolute_limit ? relative_limit : absolute_limit;
      if (strict_regression == 1) {
        printf("error: benchmark regression %s (%.3f > %.3f)\n", name, curr[name], limit) > "/dev/stderr";
        fail = 1;
      } else {
        printf("warning: benchmark regression %s (%.3f > %.3f)\n", name, curr[name], limit) > "/dev/stderr";
      }
    }
  }
  if (scoped && compared == 0) {
    print "error: no benchmark entries matched selected suite" > "/dev/stderr";
    fail = 1;
  }
  if (!scoped) {
    for (name in base) {
      if (name in curr) {
        continue;
      }
      arch = named_arch(name);
      if (arch != "" && arch != host_arch) {
        # Foreign-arch row: the runner does not emit it on this host, so its
        # absence is expected, not a gate failure. Every host-producible row
        # remains required by the branch above.
        printf("note: skipping foreign-arch baseline entry %s (host_arch=%s)\n", name, host_arch) > "/dev/stderr";
        continue;
      }
      print "error: missing benchmark entry for " name > "/dev/stderr";
      fail = 1;
    }
  }
  exit fail;
}
