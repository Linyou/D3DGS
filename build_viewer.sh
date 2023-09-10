cd SIBR_viewers
# rm -r build
cmake -Bbuild . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target install 