nvcc try.cu -I/usr/lib/x86_64-linux-gnu/openmpi/include  -lmpi
sudo env PATH="$PATH" nsys profile \
    --sample=cpu --backtrace=fp \
    --trace='cuda,nvtx' --cudabacktrace=all \
    --output=./test --force-overwrite=true \
    mpirun --allow-run-as-root -np 8 ./a.out