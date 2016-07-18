
#pragma once

#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <map>
#include <vector>
#include <limits>
#include <vector_types.h>
#include <boost/thread/mutex.hpp>
#include <boost/thread/thread.hpp>
#include <boost/thread/condition_variable.hpp>
#include "./TSDF/ColorVolume.h"
#include "./TSDF/TSDFVolume.h"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/vtk_io.h>

#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/surface/gp3.h>
#include <pcl/io/pcd_io.h>

#include <pcl/visualization/cloud_viewer.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/bilateral.h>
#include <pcl/features/normal_3d.h>
#include <pcl/surface/gp3.h>
#include <pcl/surface/mls.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>

#include <pcl/console/time.h>

#include <stack>
#include <ctime>
#include <sstream>

#include "./DataStructure/FrameInfo.h"


#define MAX_DEPTH (6.5534f)


class EvoRec
{
public:
	  EvoRec(TsdfVolume& _tsdfV, ColorVolume& _colorV, const int _width, const int _height,
		  const float _depthThres = 2.0f,  bool _doVoxelFilter = false);

	  ~EvoRec();

	  // some configurations
	  void setDepthThreshold(const float _depthThres);
	  // void setAbsIdepthVarThreshold(const float _idepthAbsVarThres);
	  
	  // process single key frame
	  //void processKeyFrame(std::string depthFile, std::string colorFile,  const FrameInfo& frameInfo);
	  void processKeyFrame(std::string depthFile, std::string colorFile, const FrameInfo& frameInfo);
	  void generatePointCloud(std::string depthFile, std::string colorFile,  const FrameInfo& frameInfo);

	  void savePointCloud(std::string outputFileName);
	  // visualize the reconstructed point cloud
	  // visualizePointCloud();

	  TsdfVolume& tsdfV;
	  ColorVolume& colorV;

private:
	float depthThres;

	int width;
	int height;
	bool doVoxelFilter;

	// reconstructed cloud
	pcl::PointCloud<pcl::PointXYZRGB>* cloudRec;
	pcl::PointCloud<pcl::PointXYZRGB>* cloudRecFiltered;
	
	// cpu type data
	unsigned short * depthData;
	PixelRGB* imageData;


	// gpu type container
	DeviceArray2D<float> depthRawScaled_;
	DeviceArray2D<unsigned short> depthRaw_;
	//DeviceArray2D<float> depthRaw_;
	DeviceArray2D<PixelRGB> color_;

	DeviceArray<pcl::PointXYZRGB> cloud_buffer_;
	DeviceArray<pcl::PointXYZRGB> cloud_device_;
};
