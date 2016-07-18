cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

project(EvoRec)


#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/Sophus/)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR})

message(${PROJECT_SOURCE_DIR})

find_package(PCL 1.6 REQUIRED)

find_package(OpenCV REQUIRED)
FIND_PACKAGE(CUDA REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(
${PCL_INCLUDE_DIRS}
${PROJECT_SOURCE_DIR}/TSDF
${PROJECT_SOURCE_DIR}/Cuda
${PROJECT_SOURCE_DIR}/Cuda/containers
${PROJECT_SOURCE_DIR}/DepthMap
${PROJECT_SOURCE_DIR}/DataStructure
#${CUDA_INCLUDE_DIRS}
)


set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode;arch=compute_50,code=sm_50")

CUDA_ADD_LIBRARY(EvoRecCuda 
	./Cuda/bilateral_pyrdown.cu
	./Cuda/extract.cu
	./Cuda/image_generator.cu
	./Cuda/maps.cu
	./Cuda/ray_caster.cu
	./Cuda/reduce.cu
	./Cuda/tsdf_volume.cu
	SHARED)
set(EvoRec_SOURCE_FILES
  ${PROJECT_SOURCE_DIR}/DepthMap/DenseMap.cpp
  ${PROJECT_SOURCE_DIR}/DepthMap/ROF.cpp
  ${PROJECT_SOURCE_DIR}/EvoRec.cpp
  ${PROJECT_SOURCE_DIR}/TSDF/ColorVolume.cpp
  ${PROJECT_SOURCE_DIR}/TSDF/TSDFVolume.cpp
  ${PROJECT_SOURCE_DIR}/Cuda/containers/device_memory.cpp
  ${PROJECT_SOURCE_DIR}/Cuda/containers/initialization.cpp
)


add_library(myLIB ${EvoRec_SOURCE_FILES})


link_directories(${PCL_LIBRARY_DIRS})

add_definitions(${PCL_DEFINITIONS})

CUDA_ADD_EXECUTABLE(EvoRec main.cpp)


target_link_libraries(EvoRec 
myLIB
${PCL_LIBRARIES}
${OpenCV_LIBS}
EvoRecCuda
)

