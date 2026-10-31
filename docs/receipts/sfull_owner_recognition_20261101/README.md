# Independent S-full owner recognition review

The previous sweeper recognises runner_multi.py directly. The admitted data-only controller invokes that runner in its own process, so the live command line names the controller instead. The retained old source returns false for the actual live controller; the fixed source returns true.

The added recognition is limited to the exact reviewed controller path and source hash, exact argv, the pinned manifest and complete source inventory, the same state directory, and a hash-anchored root admission. Launch identity is rechecked at its admitted time, so a live process still owns its resources during deadline-triggered stop and relay. This does not authorise further runtime or alter a cap.

The portable synthetic-controller tests and the separate actual frozen-controller matrix retain positive and negative controls. Unknown controllers, dead or empty-command processes, source/manifest/inventory drift, wrong state, malformed argv, missing/changed anchors and unapproved admissions are refused. The existing ordinary runner path remains unchanged.

This review invokes no provider accessor or mutation and changes no driver or watchdog configuration. The fixed public source and the actual watchdog copy were byte-identical at review. Raw logs, source hashes and the real-binding test snapshot are retained. The snapshot is stored as .py.txt so it cannot be collected as a CI test requiring local campaign artifacts.
