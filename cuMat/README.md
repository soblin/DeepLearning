# cuMat

C++ Matrix library using CUDA

```bash
cmake . && make
```
creates libcumat.so

cublasSgeam computes C <= alpha*op(A) + beta*op(B)
cublasSgemm computes C <= alpah*op(A)@op(B) + beta*C

member functions

copy(const cuMat &A) this <= A

ones()   this[][] <= 1

void plus(const cuMat &B, cuMat &C)            C[i][j] <= this[i][j] + B[i][j]

void plus(const float b, cuMat &C)             C[i][j] <= this[i][j] + b

void plus(const float b, cuMat &C, cuMat &D)   D[i][j] <= this[i][j] + b * C[i][j]

void minus(const cuMat &B, cuMat &C)           C[i][j] <= this[i][j] - B[i][j]

void mul(const float b, cuMat &C)              C[i][j] <= b * this[i][j]

void mul(const cuMat &A, cuMat &B)             B[i][j] <= this[i][j] * A[i][j]

void mul_plus(const float b, cuMat &C)         C[i][j] += b * this[i][j]

void mul_plus(const cuMat &A, cuMat &B,
              const float a, const float b)    B[i][j] += a * this[i][j] * beta * A[i][j]

cuMat dot(const cuMat &A)                      return this@A

void dot(const cuMat &B, cuMat &C)             C <= this@B

void dot_plus(const cuMat &B, cuMat &C)        C += this@B

void transpose_dot_plus(const cuMat &B,
                        cuMat &C)              C += trans(this)@B
                        
void dot_transpose_plus(const cuMAt &B,
                        cuMat &C)              C += this@trans(B)
                        
void transpose(void)                           this <= trans(this)

void plus_util(float alpha, float beta,
               cuMat &B, cuMat &C)             C <= alpha*this + beta*B

cuMat log(void)                                return log(this[i][j])

cuMat sqrt(void)                               return sqrt(this[i][j])

cuMat sqrt_d(void)                             return (1/2*sqrt(this[i][j])) ( == d/dx sqrt(x))