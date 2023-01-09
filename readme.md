# Quadratic Surface Visualization Using Ray-casting and Polygonization

For polygonization, we'd like to use [Marching Cubes](https://dl.acm.org/doi/abs/10.1145/37402.37422).

For ray-casting, we'd like to use [GPU-Based Ray-Casting of Quadratic Surfaces](https://reality.cs.ucl.ac.uk/projects/quadrics/pbg06.html).

## Usage

This project is written in C++ along with glsl shaders.

1. Clone the project (or just use existing source)
    ```shell
    git clone https://github.com/dendenxu/quadratic_surface --recursive
    ```
    If you've already got the packaged source files, this step can be omitted.
2. Run cmake to configure and build the project.

    ```shell
    # Windows
    mkdir build
    cd build
    cmake ..
    cmake --build . --config Realease -j

    # Linux
    mkdir build
    cd build
    cmake ..
    make -j
    ```

    During `cmake ..`, you might see prompts about missing system packages, install them to build the program properly.

## Quadrics

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

-   We implemented a shader in `shaders/rt.frag` to trace rays and intersect with the defined quadrics.
-   We perform the Marching Cubes algorithm on the implicit surface in `include/volrend/quadric.hpp`.
