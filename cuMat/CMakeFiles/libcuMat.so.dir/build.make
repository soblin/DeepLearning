# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/mamoru/DeepLearning/cuMat

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/mamoru/DeepLearning/cuMat

# Include any dependencies generated for this target.
include CMakeFiles/libcuMat.so.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/libcuMat.so.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/libcuMat.so.dir/flags.make

CMakeFiles/libcuMat.so.dir/libcuMat.so_generated_mat_ones_kernel.cu.o: CMakeFiles/libcuMat.so.dir/libcuMat.so_generated_mat_ones_kernel.cu.o.depend
CMakeFiles/libcuMat.so.dir/libcuMat.so_generated_mat_ones_kernel.cu.o: CMakeFiles/libcuMat.so.dir/libcuMat.so_generated_mat_ones_kernel.cu.o.cmake
CMakeFiles/libcuMat.so.dir/libcuMat.so_generated_mat_ones_kernel.cu.o: mat_ones_kernel.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/mamoru/DeepLearning/cuMat/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building NVCC (Device) object CMakeFiles/libcuMat.so.dir/libcuMat.so_generated_mat_ones_kernel.cu.o"
	cd /home/mamoru/DeepLearning/cuMat/CMakeFiles/libcuMat.so.dir && /usr/local/bin/cmake -E make_directory /home/mamoru/DeepLearning/cuMat/CMakeFiles/libcuMat.so.dir//.
	cd /home/mamoru/DeepLearning/cuMat/CMakeFiles/libcuMat.so.dir && /usr/local/bin/cmake -D verbose:BOOL=$(VERBOSE) -D build_configuration:STRING= -D generated_file:STRING=/home/mamoru/DeepLearning/cuMat/CMakeFiles/libcuMat.so.dir//./libcuMat.so_generated_mat_ones_kernel.cu.o -D generated_cubin_file:STRING=/home/mamoru/DeepLearning/cuMat/CMakeFiles/libcuMat.so.dir//./libcuMat.so_generated_mat_ones_kernel.cu.o.cubin.txt -P /home/mamoru/DeepLearning/cuMat/CMakeFiles/libcuMat.so.dir//libcuMat.so_generated_mat_ones_kernel.cu.o.cmake

# Object files for target libcuMat.so
libcuMat_so_OBJECTS =

# External object files for target libcuMat.so
libcuMat_so_EXTERNAL_OBJECTS = \
"/home/mamoru/DeepLearning/cuMat/CMakeFiles/libcuMat.so.dir/libcuMat.so_generated_mat_ones_kernel.cu.o"

liblibcuMat.so.so: CMakeFiles/libcuMat.so.dir/libcuMat.so_generated_mat_ones_kernel.cu.o
liblibcuMat.so.so: CMakeFiles/libcuMat.so.dir/build.make
liblibcuMat.so.so: /usr/local/cuda-9.1/lib64/libcudart.so
liblibcuMat.so.so: CMakeFiles/libcuMat.so.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/mamoru/DeepLearning/cuMat/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared library liblibcuMat.so.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/libcuMat.so.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/libcuMat.so.dir/build: liblibcuMat.so.so

.PHONY : CMakeFiles/libcuMat.so.dir/build

CMakeFiles/libcuMat.so.dir/requires:

.PHONY : CMakeFiles/libcuMat.so.dir/requires

CMakeFiles/libcuMat.so.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/libcuMat.so.dir/cmake_clean.cmake
.PHONY : CMakeFiles/libcuMat.so.dir/clean

CMakeFiles/libcuMat.so.dir/depend: CMakeFiles/libcuMat.so.dir/libcuMat.so_generated_mat_ones_kernel.cu.o
	cd /home/mamoru/DeepLearning/cuMat && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/mamoru/DeepLearning/cuMat /home/mamoru/DeepLearning/cuMat /home/mamoru/DeepLearning/cuMat /home/mamoru/DeepLearning/cuMat /home/mamoru/DeepLearning/cuMat/CMakeFiles/libcuMat.so.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/libcuMat.so.dir/depend
