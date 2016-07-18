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
#include <sstream>

class ROF
{
public:
	ROF(int _width, int _height, int _maxIter = 1000, float _lambda = 1.0f, float epsilon = 0.0002f);
	~ROF();

	void setMask(bool* _pixelMask);
	void setWeight(float* _pixelWeight);
	void setGradientWeight(float* _pixelGradientWeight);

	void denoise(unsigned char* dst, const float* src);
	void denoise(unsigned char* dst, const unsigned char* src);
	void denoise(float* dst, const float* src);
	void denoise(float* dst, const unsigned char* src);

	void convertFloat2UnsignedChar(unsigned char* dst, const float* src);
	void convertUnsignedChar2Float(float* dst, const unsigned char* src);

private:
	int width;
	int height;
	float lambda;
	float epsilon;
	int maxIter;
	float sigmaQ;
	float sigmaD;
	float gamma;

	bool* pixelMask;
	float* pixelWeight;
	float* pixelGradientWeight;

	float* qCurrent;
	float* qNext;
	float* dCurrent;
	float* dNext;
	float* aCurrent;
	float* aNext;

	void initialize();
	void updateADDenoise();
	void updateAtQDenoise();
	void updateDQ();
	void updateSigmaDQ();
	void finalise(float* dst);

	float computeDual();
	float computePrimal();
};
