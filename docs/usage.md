# Usage

```c++
size_t vec_size = 256;
float *x = new float[vec_size];

/*
** Fill x with something.
** ...
*/

/* Create vector on GPU. */
Vector d_x(vec_size, x);

/* Create another vector. */
Vector d_y(vec_size, y);

/* Add two vectors. */
Vector c(vec_size);
float elapsed = VectorMul(blockSize, a, b, c);
std::cout << "Elapsed " << elapsed << " ms" << std::endl;

/* Fetch vector. */
float *r = new float[vec_size];
c.Fetch(vec_size, r);
```
