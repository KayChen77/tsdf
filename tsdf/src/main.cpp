#include <iostream>
#include <fstream>
#include <cv.h>
#include <highgui.h>
#include <dirent.h>
#include <string.h>
#include "EvoRec.h"
#include "./DepthMap/DenseMap.h"
 
int main (int argc, char** argv) 
{
	const Eigen::Vector3i resolution = Eigen::Vector3i(512, 512, 512);
	//const Eigen::Vector3f volumeSize = Eigen::Vector3f(0.24f, 0.24, 0.24f);
	const Eigen::Vector3f volumeSize = Eigen::Vector3f(4.0f, 4.0f, 4.0f);
	int width = 512;
	int height = 424;
	float depthThres = 2.0f;
	bool doVoxelFilter = false;

	TsdfVolume tsdfV(resolution, volumeSize);
	ColorVolume colorV(tsdfV);
	EvoRec evoRec(tsdfV, colorV, width, height, depthThres, doVoxelFilter);
	
	// begins process evoslam data
	FrameInfo Kf1;
	Kf1.rotMatUnscaled = Eigen::Matrix3f::Identity();
	/*Kf1.rotMatUnscaled(0, 0) = 1;
	Kf1.rotMatUnscaled(0, 1) = 0.0215;
	Kf1.rotMatUnscaled(0, 2) = 0;
	Kf1.rotMatUnscaled(1, 0) = 0;
	Kf1.rotMatUnscaled(1, 1) = 0.9877;
	Kf1.rotMatUnscaled(1, 2) = 0.1546;
	Kf1.rotMatUnscaled(2, 0) = 0;
	Kf1.rotMatUnscaled(2, 1) = -0.1546;
	Kf1.rotMatUnscaled(2, 2) = 0.9877;
	Kf1.transVec = Eigen::Vector3f(-0.05, -0.28, -0.93);*/
	Kf1.transVec = Eigen::Vector3f(0, 0, 0);
	//Kf1.transVec = Eigen::Vector3f(1.5, 1.5, 1.5);
	Kf1.scale = 1;
	Kf1.intr = Intr(367.221, 367.221, 252.952, 208.622);

	FrameInfo Kf2;
	Kf2.rotMatUnscaled = Eigen::Matrix3f::Identity();
	Kf2.rotMatUnscaled(0, 0) = 0.134495157429655;
	Kf2.rotMatUnscaled(0, 1) = 0.080498653293076;
	Kf2.rotMatUnscaled(0, 2) = -0.987639113971279;
	Kf2.rotMatUnscaled(1, 0) = -0.144135633421401;
	Kf2.rotMatUnscaled(1, 1) = 0.987683785283996;
	Kf2.rotMatUnscaled(1, 2) = 0.060874128045401;
	Kf2.rotMatUnscaled(2, 0) = 0.980375423909730;
	Kf2.rotMatUnscaled(2, 1) = 0.134166713849142;
	Kf2.rotMatUnscaled(2, 2) = 0.144441410574456;
	Kf2.rotMatUnscaled = Kf1.rotMatUnscaled * Kf2.rotMatUnscaled;
	Kf2.transVec = Kf1.rotMatUnscaled * (Eigen::Vector3f(1.128942737114943e+03 / 1000.0f, -1.432967440937596e+02 / 1000.0f, 7.893658796324055e+02 / 1000.0f)) + Kf1.transVec;
	Kf2.scale = 1;
	Kf2.intr = Intr(366.507, 366.507, 259.864, 206.676);

	FrameInfo Kf3;
	//Kf.rotMatUnscaled << 1, 0, 0, 0, -1, 0, 0, 0, 1;
	//Kf.transVec  = Eigen::Vector3f(0,0,0);
	Kf3.rotMatUnscaled = Eigen::Matrix3f::Identity();
	Kf3.rotMatUnscaled(0, 0) = -0.993943048364607;
	Kf3.rotMatUnscaled(0, 1) = 0.030325927333675;
	Kf3.rotMatUnscaled(0, 2) = -0.105629327078354;
	Kf3.rotMatUnscaled(1, 0) = 0.004398346757531;
	Kf3.rotMatUnscaled(1, 1) = 0.971379193082247;
	Kf3.rotMatUnscaled(1, 2) = 0.237493405787788;
	Kf3.rotMatUnscaled(2, 0) = 0.109808338269340;
	Kf3.rotMatUnscaled(2, 1) = 0.235590325306951;
	Kf3.rotMatUnscaled(2, 2) = -0.965628980234278;
	Kf3.rotMatUnscaled = Kf1.rotMatUnscaled * Kf3.rotMatUnscaled;
	Kf3.transVec = Kf1.rotMatUnscaled * (Eigen::Vector3f(2.507968161837474e+02 / 1000.0f, -3.275127035049086e+02 / 1000.0f, 2.057818523590235e+03 / 1000.0f) )+ Kf1.transVec;
	Kf3.scale = 1;
	Kf3.intr = Intr(366.996, 366.996, 256.466, 209.012);

	FrameInfo Kf4;
	Kf4.rotMatUnscaled = Eigen::Matrix3f::Identity();
	Kf4.rotMatUnscaled(0, 0) = -0.073896009032637;
	Kf4.rotMatUnscaled(0, 1) = -0.155250763238798;
	Kf4.rotMatUnscaled(0, 2) = 0.985107395344700;
	Kf4.rotMatUnscaled(1, 0) = 0.158951733970376;
	Kf4.rotMatUnscaled(1, 1) = 0.973346494691277;
	Kf4.rotMatUnscaled(1, 2) = 0.165320741408982;
	Kf4.rotMatUnscaled(2, 0) = -0.984517001436166;
	Kf4.rotMatUnscaled(2, 1) = 0.168801071637521;
	Kf4.rotMatUnscaled(2, 2) = -0.047249043346561;
	Kf4.rotMatUnscaled = Kf1.rotMatUnscaled * Kf4.rotMatUnscaled;
	Kf4.transVec = Kf1.rotMatUnscaled * ( Eigen::Vector3f(-8.272323915033728e+02 / 1000.0f, -1.737204056866098e+02 / 1000.0f, 1.256884341974619e+03 / 1000.0f)) + Kf1.transVec;
	Kf4.scale = 1;
	Kf4.intr = Intr(366.604, 366.604, 257.112, 198.123);

	evoRec.processKeyFrame("test_tsdf_data_4/022718143547_undistorted_1_1.bin","test_tsdf_data_4/022718143547_registered_1_1.jpg", Kf1);
	//evoRec.processKeyFrame("test_tsdf_data_4/023446243547_undistorted_1_1.bin", "test_tsdf_data_4/023446243547_registered_1_1.jpg", Kf2);
	//evoRec.processKeyFrame("test_tsdf_data_4/001489661447_undistorted_1_1.bin", "test_tsdf_data_4/001489661447_registered_1_1.jpg", Kf3);
	//evoRec.processKeyFrame("test_tsdf_data_4/001496161447_undistorted_1_1.bin", "test_tsdf_data_4/001496161447_registered_1_1.jpg", Kf4);

	evoRec.savePointCloud("tsdf_cloud.pcd");
	//evoRec.generatePointCloud("test_tsdf_data_4/022718143547_undistorted_1_1.bin","test_tsdf_data_4/022718143547_registered_1_1.jpg", Kf1);

}
