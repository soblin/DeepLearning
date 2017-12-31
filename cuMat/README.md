# cuMat

C++ Matrix library using CUDA

```bash
cmake . && make
```
creates libcumat.so

member functions

copy(const cuMat &A) this <= A

ones()   this[][] <= 1

void plus(const cuMat &B, cuMat &C)            C[][] <= this[][] + B[][]
void plus(const float b, cuMat &C)             C[][] <= this[][] + b
void plus(const float b, cuMat &C, cuMat &D)   D[][] <= this[][] + b * C[][]    
void minus(const cuMat &B, cuMat &C)           C[][] <= this[][] - B[][]

void mul(const float b, cuMat &C)              C[][] <= b * this[][]
void mul(const cuMat &A, cuMat &B)             B[][] <= this[][] * A[][]
void mul_plus(const float b, cuMat &C)         C[][] += b * this[][]
void mul_plus(const cuMat &A, cuMat &B, const float a, const float b)
                                          B[][] += a * this[][] * beta * A[][]

cuMat dot(const cuMat &A)                       return this@A
void dot(const cuMat &B, cuMat &C)                 C <= this@B
void dot_plus(const cuMat &B, cuMat &C)            C += this@B
