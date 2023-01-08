# Quadratic Surface Visualization Using Ray-casting and Polygonization

For polygonization, we'd like to use CUDA marching cubes.

For ray-casting, we'd like to use [GPU-Based Ray-Casting of Quadratic Surfaces](https://reality.cs.ucl.ac.uk/projects/quadrics/pbg06.html).

## Ray-casting

Quatrics are defined with variables `A-J`:

In general, quadratic surfaces are defined as the set of roots of a polynomial of degree two:

$$
f(x,y,z) = Ax^2 + 2Bxy + 2Cxz + 2Dx + Ey^2 + 2Fyz + Gy + Hz^2 + 2Iz + J = 0
$$

Using homogeneous coordinates $\mathbf{x}=(x,y,z,1)^T$ the quadratic surface can be compactly written as: $\mathbf{x}^T\mathbf{Q}\mathbf{x}=0$ with $Q$:

$$
\mathbf{Q} =
\begin{bmatrix}
A & B & C & D \\
B & E & F & G \\
C & F & H & I \\
D & G & I & J
\end{bmatrix}
$$
10 variables in total to define one quadrics.

We'd like to support multiple quadrics for rendering, we kind of need to define a new file format?

Just do not try to make your life harder...

How to pass data with varying length to the GPU?
