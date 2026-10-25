set -e
SAW=/mnt/c/ml_artifacts/score_audit/work
mkdir -p ~/sa_src_d53c ~/sa_build_d53c
cd ~/sa_src_d53c
[ -f CMakeLists.txt ] || tar -xf $SAW/sa_src_d53c.tar
cd ~/sa_build_d53c
[ -f CMakeCache.txt ] || cmake ~/sa_src_d53c -DCMAKE_BUILD_TYPE=Release -DMICROTORCH_CUDA=OFF > cfg.log 2>&1
make -j${SA_JOBS:-2} mtstudio > make.log 2>&1
ls -la ~/sa_build_d53c/mtstudio
