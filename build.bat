nvcc hello_world.cu -o hello_world.exe
nvcc vector_add.cu -o vector_add.exe
nvcc matrix_add_1d.cu -o matrix_add_1d.exe
nvcc matrix_add.cu -o matrix_add.exe
nvcc matrix_multiply_naive.cu -o matrix_multiply_naive.exe -lcublas
nvcc matrix_multiply_tiled.cu -o matrix_multiply_tiled.exe -lcublas

:: Use -Xptxas to print kernel register count
nvcc -Xptxas -v matrix_multiply_coarse_vectorized.cu -o matrix_multiply_coarse_vectorized.exe -lcublas