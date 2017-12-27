file(REMOVE_RECURSE
  "libmat_ones_kernel.cu.pdb"
  "libmat_ones_kernel.cu.so"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/mat_ones_kernel.cu.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
