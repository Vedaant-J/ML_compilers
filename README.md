# mp1
Machine Project 1

### Compiling Part 1 (CPU) on Apple devices:
Note: Please pass the -O3 flag below only for the last optimization as also suggested in the mp1 handout.

```
brew install llvm libomp
export PATH="/opt/homebrew/opt/llvm/bin:$PATH"
export LDFLAGS="-L/opt/homebrew/opt/llvm/lib"
export CPPFLAGS="-I/opt/homebrew/opt/llvm/include"
export CXXFLAGS="-fopenmp"
cd mp1
mkdir build
cd build
cmake ..
clang++ -O3 -march=native -ffast-math -fopenmp ../cpu/gemm_cpu.cpp -L/opt/homebrew/opt/llvm/lib  -o mp1_cpu 
```