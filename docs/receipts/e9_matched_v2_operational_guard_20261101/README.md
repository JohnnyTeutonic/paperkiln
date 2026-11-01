# E9 v2 independent operational admission review

Reviewed root-prepared source c9b82fd6718497a4ace4fba62f6e23b8ed2346a7 and original guard 334732a05a5bcb7df46f74795202f05b7cd6cc9e163b28e71bb107d9bb7e94b3. This folder is separate from root preparation and Claude's files. No provider operation, GPU launch, watchdog configuration edit, or root prepared-file edit occurred.

## Findings

The original guard verifies the signed-review hash, argv equality, numerical cap flags, absolute deadline future-ness, upload inventory and pinned file bytes. However, original `validate()` accepts six adversarial prospective reviews demonstrated in tests: pre-existing final artifacts; non-home state with a foreign volume; `--commit` different from frozen runner source; absolute deadline more than two hours after admission; a symlink dependency pin; and an upload source absent from pins. Because the root review itself is hash-bound, these are missing review-construction invariants rather than untrusted-caller bypasses. They matter at admission preparation.

The proposal explicitly binds frozen runner/source/jobs/boot/art paths and upload source pins, requires canonical regular dependency files, caps the absolute window at two hours, and requires a hash-bound fresh-path attestation. At first real execution it creates an admission anchor; the same review can restart using its own state, while a foreign anchor or dirty initial path is refused. Owner probes do not create anchors or claim real admission.

`prepare_dispatch.py` is an offline helper for root adoption. It must run at actual dispatch, using a freshly collected root serial-check receipt; it checks no GPU campaign active, preceding campaign home and owned resources deleted, verifies every frozen pin, refuses pre-existing state/artifact paths and sets the deadline to now plus two hours. It writes review/dispatch records only; it neither checks provider state itself nor enables the watchdog. The serial snapshot must contain `checked_at_utc`, `gpu_campaigns_active: 0`, `previous_campaign_home: true`, `previous_owned_resources_deleted: true`; root is responsible for collecting those facts truthfully. Do not fabricate a serial check from this review's synthetic test fixture.

## Validation

16 tests pass, including a live <=30-second no-provider probe executing the proposed guard with real frozen source/input pins and actual frozen `runpod_sweep.runner_alive`. The probe's PID is recognized and an absent PID is refused. Actual sweeper fallback recognizes the embedded runner argv; this test establishes operational ownership recognition, not strengthened independent guard hashing inside the sweeper.

The original queued plan stays unadmitted. Full GPU queue ordering and current provider/account checks remain root's responsibility. Frozen v2 checker, scientific margins, failure evidence and packet files are unchanged. Both final gate flags and local replay are still required. `validation.json` binds tested sources and outcomes.

## Suggested dispatch preparation (root only)

Run `python3 prepare_dispatch.py --plan /home/jonat/e9_matched_v2_admission_20261101/plan.json --serial-check <fresh-root-serial-receipt.json> --output <new-dispatch-folder>`. Default guard is the separately proposed guard here. Review output and use its exact guard argv through the existing watchdog; do not start a second driver.
