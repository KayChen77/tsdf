#pragma once

#include <iostream>
#include <fstream>
#include <cv.h>
#include <highgui.h>
#include <dirent.h>
#include <string.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/registration/icp.h>
#include <pcl/io/ply_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/time.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/io/vtk_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>
#include <pcl/surface/gp3.h>
#include <pcl/surface/mls.h>
#include <pcl/surface/grid_projection.h>

#include <pcl/ModelCoefficients.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/console/time.h>
#include <stack>
#include <ctime>
#include <time.h>
#include <sstream>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/surface/marching_cubes_rbf.h>
#include <pcl/surface/marching_cubes_hoppe.h>
#include <pcl/surface/poisson.h>

#include <pcl/filters/bilateral.h>

#include "../DataStructure/FrameInfo.h"
#include "ROF.h"

#define DIVISION_EPS (1e-10f)

class DenseMap
{
public:
	DenseMap(int _width, int _height, bool _doBilateralFilter = false, const float _idepthMax = 4.0f, 
		const float _idepthMin = 0.0f, const int _fillHoleTestNum = 100, const float _idepthVarMax = 0.8f, const float _maxIntensityError = 5.0f);
	~DenseMap();

	void processEvoSlamData(std::string inKFFileName, std::string ImageFileName, std::string outKFFileName);

private:
	int width;
	int height;
	float idepthMax;
	float idepthMin;
	int fillHoleTestNum;
	bool doBilateralFilter;
	float idepthVarMax;
	float maxIntensityError;

	void fillHolesVar(float* iDepthVarKF, float defaultVar);

	int getdir(std::string dir, std::vector<std::string> &files);

	void setDepthImage(unsigned char* imageIdepth, const float* idepth);
	void loadKFData(std::string fileName, std::vector<FrameInfo>& KFInfo);

	int getDir(std::string dir, std::vector<std::string> &files);

	void fillHolesCostMin(const float* idepthOrg, float* idepthFilled, float* idepthVar, bool* idepthMask,
		const unsigned char* imageKF, const unsigned char* imageStereo, Eigen::Matrix3f& KF2F_rotMat, Eigen::Vector3f& KF2F_transVec);

	void getDepthWeight(const float* idepthVarKF, float* idepthWeight);
	void getPixelGradientWeight(const unsigned char* image, float* pixelGradientWeight);

	float getInterpolatedElement(const unsigned char* image, const float x, const float y);
	int toCamera(const Eigen::Vector3f& Wxp);
	Eigen::Vector2f KF2F(int x, int y, float idepth, Eigen::Matrix3f& KF2F_rotMat, Eigen::Vector3f& KF2F_transVec);
};
