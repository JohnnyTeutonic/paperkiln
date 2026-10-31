# Required CMake asset before any numerical launch

Root's clean extraction/configure rejected the first prospective archive because
the unchanged CMake file reads `studio/chat.html`, which the archive omitted.
No GPU or paid job ran. The failed preflight is retained in
`archive_preflight_v1_failure.json`.

V2 adds only that file from the same frozen source commit. Parent comparison
verifies every previous file payload is unchanged and the sole additional path
is `studio/chat.html`. The updated manifest is `inputs_v2.json`. Clean extraction,
all source/hash checks and CPU CMake configure now pass (`archive_preflight_v2.json`).
The relative command job is bound to the new manifest. Numerical tests, fixture,
old source, implementation, parser and bounds are unchanged.

Full-object build, actual L4 execution, transport/ownership binding and landing
replay remain pending. This preflight does not assert CUDA execution or success.
