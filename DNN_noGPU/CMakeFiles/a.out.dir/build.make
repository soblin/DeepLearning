# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.3

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
CMAKE_COMMAND = /opt/Xilinx/SDK/2017.3/tps/lnx64/cmake-3.3.2/bin/cmake

# The command to remove a file.
RM = /opt/Xilinx/SDK/2017.3/tps/lnx64/cmake-3.3.2/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/mamoru/DeepLearning/DNN_noGPU

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/mamoru/DeepLearning/DNN_noGPU

# Include any dependencies generated for this target.
include CMakeFiles/a.out.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/a.out.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/a.out.dir/flags.make

CMakeFiles/a.out.dir/function.cc.o: CMakeFiles/a.out.dir/flags.make
CMakeFiles/a.out.dir/function.cc.o: function.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/mamoru/DeepLearning/DNN_noGPU/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/a.out.dir/function.cc.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/a.out.dir/function.cc.o -c /home/mamoru/DeepLearning/DNN_noGPU/function.cc

CMakeFiles/a.out.dir/function.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/a.out.dir/function.cc.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/mamoru/DeepLearning/DNN_noGPU/function.cc > CMakeFiles/a.out.dir/function.cc.i

CMakeFiles/a.out.dir/function.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/a.out.dir/function.cc.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/mamoru/DeepLearning/DNN_noGPU/function.cc -o CMakeFiles/a.out.dir/function.cc.s

CMakeFiles/a.out.dir/function.cc.o.requires:

.PHONY : CMakeFiles/a.out.dir/function.cc.o.requires

CMakeFiles/a.out.dir/function.cc.o.provides: CMakeFiles/a.out.dir/function.cc.o.requires
	$(MAKE) -f CMakeFiles/a.out.dir/build.make CMakeFiles/a.out.dir/function.cc.o.provides.build
.PHONY : CMakeFiles/a.out.dir/function.cc.o.provides

CMakeFiles/a.out.dir/function.cc.o.provides.build: CMakeFiles/a.out.dir/function.cc.o


# Object files for target a.out
a_out_OBJECTS = \
"CMakeFiles/a.out.dir/function.cc.o"

# External object files for target a.out
a_out_EXTERNAL_OBJECTS =

a.out: CMakeFiles/a.out.dir/function.cc.o
a.out: CMakeFiles/a.out.dir/build.make
a.out: CMakeFiles/a.out.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/mamoru/DeepLearning/DNN_noGPU/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable a.out"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/a.out.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/a.out.dir/build: a.out

.PHONY : CMakeFiles/a.out.dir/build

CMakeFiles/a.out.dir/requires: CMakeFiles/a.out.dir/function.cc.o.requires

.PHONY : CMakeFiles/a.out.dir/requires

CMakeFiles/a.out.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/a.out.dir/cmake_clean.cmake
.PHONY : CMakeFiles/a.out.dir/clean

CMakeFiles/a.out.dir/depend:
	cd /home/mamoru/DeepLearning/DNN_noGPU && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/mamoru/DeepLearning/DNN_noGPU /home/mamoru/DeepLearning/DNN_noGPU /home/mamoru/DeepLearning/DNN_noGPU /home/mamoru/DeepLearning/DNN_noGPU /home/mamoru/DeepLearning/DNN_noGPU/CMakeFiles/a.out.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/a.out.dir/depend

