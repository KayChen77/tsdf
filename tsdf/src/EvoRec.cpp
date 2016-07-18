#include <fstream>
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/io/io.h>
#include <algorithm>
#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/Cholesky>
#include <Eigen/Geometry>
#include <Eigen/LU>
#include <boost/bind.hpp>
#include <boost/thread.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <boost/filesystem.hpp>
#include "EvoRec.h"



EvoRec::EvoRec(TsdfVolume& _tsdfV, ColorVolume& _colorV, const int _width, const int _height,
	const float _depthThres,  bool _doVoxelFilter)
	: tsdfV(_tsdfV), 
	colorV(_colorV), 
	width(_width), 
	height(_height), 
	depthThres(_depthThres), 
	doVoxelFilter(_doVoxelFilter)
{

	cloudRec = new pcl::PointCloud<pcl::PointXYZRGB>();
	cloudRecFiltered = new pcl::PointCloud<pcl::PointXYZRGB>();

	depthData = new unsigned short[width*height];
	imageData = new PixelRGB[width*height];

	depthRawScaled_.create(height, width);
	depthRaw_.create(height, width);
	color_.create(height, width);
}

EvoRec::~EvoRec()
{
}

void EvoRec::setDepthThreshold(const float _depthThres)
{
	depthThres = _depthThres;
}




//// process single key frame
//void EvoRec::processKeyFrame(std::string depthFile, std::string colorFile, const FrameInfo& frameInfo)
//{
//	Intr intr = Intr(367.221, 367.221, 252.952, 208.622);
//	std::ifstream inDepth(depthFile.c_str(),std::ios::binary);
//	float* idepth = new float[width*height];
//	inDepth.read((char*)idepth, sizeof(float)* width * height);
//
//	cv::Mat imageRGB;
//	imageRGB =  cv::imread(colorFile.c_str(), CV_LOAD_IMAGE_COLOR);
//	imageRGB.convertTo(imageRGB, CV_8UC3);
//	
//	// filter data and convert from float to unsigned short and pixelrgb
//        
//	for (int v = 0; v < height; v++)
//	{
//		for (int u = 0; u < width; u++)
//		{
//                        
//			int idx = u + v * width;
//			// filter some invalid points
//			float depth =  idepth[idx];
//
//			/*
//			if (depth > depthThres || depth > MAX_DEPTH
//				 || depth < 0)
//			{
//				depthData[idx] = 0;
//				imageData[idx].r = 0;
//				imageData[idx].g = 0;
//				imageData[idx].b = 0;
//				continue;
//			}
//			*/
//			// convert from float to short
//			depthData[idx] = (unsigned short)depth;
//			// convert from float to unsigned char
//
//			imageData[idx].b = (unsigned char)imageRGB.at<cv::Vec3b>(v,u)[0];
//			imageData[idx].g = (unsigned char)imageRGB.at<cv::Vec3b>(v,u)[1];
//			imageData[idx].r = (unsigned char)imageRGB.at<cv::Vec3b>(v,u)[2];
//			//OpenCV store pixels as BGR.
//		}
//	}
//
//	// upload cpu data to gpu memory
//	depthRaw_.upload(depthData, width*2, height, width);
//	color_.upload(imageData, width*3, height, width);
//
//	// convert from column major (default) to row major 
//	Eigen::Matrix<float, 3, 3, Eigen::RowMajor> rotMatrix;
//	for (int ix = 0; ix < 3; ix++)
//	{
//		for (int iy = 0; iy < 3; iy++)
//		{
//			rotMatrix(ix, iy) = frameInfo.rotMatUnscaled(ix, iy);
//		}
//	}
//	Eigen::Matrix<float, 3, 3, Eigen::RowMajor> rotMatrixInverse = rotMatrix.inverse();
//	Eigen::Vector3f transVec;
//	for (int ix = 0; ix < 3; ix++)
//		transVec(ix) = frameInfo.transVec(ix);
//
//	float3 device_volume_size = device_cast<const float3> (tsdfV.getSize());
//	
//	Mat33&  device_Rcam = device_cast<Mat33> (rotMatrix);
//	float3& device_tcam = device_cast<float3> (transVec);
//	Mat33&   device_Rcam_inv = device_cast<Mat33> (rotMatrixInverse);
//
//	// static tsdf volume
//	
//	int3 emptyVoxel;
//	emptyVoxel.x = 0;
//	emptyVoxel.y = 0;
//	emptyVoxel.z = 0;
//	
//	
//
//	// aggregate information to tsdf data
//	printf("aggregating information to tsdf volume ... \n");
//
//	integrateTsdfVolume(depthRaw_,
//		intr, 
//		device_volume_size,
//		device_Rcam_inv,
//		device_tcam,
//		tsdfV.getTsdfTruncDist(),
//		tsdfV.data(),
//		depthRawScaled_,
//		emptyVoxel,
//		colorV.data(),
//		color_);
//
//	
//
//	printf("extracting point cloud ... \n");
//	// extract and update the reconstructed point cloud
//	const Eigen::Vector3i volumeSize = tsdfV.getResolution();
//	cloud_buffer_ = tsdfV.fetchCloud(cloud_device_,
//		emptyVoxel,
//		colorV.data(),
//		0, volumeSize(0),
//		0, volumeSize(1),
//		0, volumeSize(2),
//		emptyVoxel);
//	/*size_t size= cloud_buffer_.size();
//	size_t size2= cloud_device_.size();
//	std::cout << "buffer size: " << size << endl;
//	std::cout << "device size: " << size2 << endl;*/
//	//DeviceArray<pcl::PointXYZRGB> newBuffer(cloud_device_.ptr(), width*height);	
//	//std::cout << "new buffer size: " << newBuffer.size () << endl;
//	
//	printf("downloading to cpu data ... \n");
//	cloud_buffer_.download(cloudRec->points);
//	//newBuffer.download(cloudRec->points);
//	//cloud_device_.download(cloudRec->points);
//	//std::cout << "rec size: " << cloudRec->points.size () << endl;
//	cloudRec->width = (int)cloudRec->points.size ();
//	cloudRec->height = 1;
//	//std::cout << "rec size after: " << cloudRec->points.size () << endl;
//	printf("KF processed ... \n ");
//    pcl::io::savePCDFile ("tsdf_cloud7.pcd", *cloudRec,false);
//		
//	inDepth.close();
//	delete idepth;
//	
//}

//// process single key frame
void EvoRec::processKeyFrame(std::string depthFile, std::string colorFile, const FrameInfo& frameInfo)
{
	std::ifstream inDepth(depthFile.c_str(),std::ios::binary);
	float* idepth = new float[width*height];
	inDepth.read((char*)idepth, sizeof(float)* width * height);

	cv::Mat imageRGB;
	imageRGB =  cv::imread(colorFile.c_str(), CV_LOAD_IMAGE_COLOR);
	//imageRGB.convertTo(imageRGB, CV_8UC3);
	
	// filter data and convert from float to unsigned short and pixelrgb
    int count = 0;    
	for (int v = 0; v < height; v++)
	{
		for (int u = 0; u < width; u++)
		{
                        
			int idx = u + v * width;
			// filter some invalid points
			float depth =  idepth[idx];

			/*
			if (depth > depthThres || depth > MAX_DEPTH
				 || depth < 0)
			{
				depthData[idx] = 0;
				imageData[idx].r = 0;
				imageData[idx].g = 0;
				imageData[idx].b = 0;
				continue;
			}
			*/
			// convert from float to short
			depthData[idx] = (unsigned short)depth;
			// convert from float to unsigned char

			imageData[idx].b = (unsigned char)imageRGB.at<cv::Vec3b>(v,u)[0];
			imageData[idx].g = (unsigned char)imageRGB.at<cv::Vec3b>(v,u)[1];
			imageData[idx].r = (unsigned char)imageRGB.at<cv::Vec3b>(v,u)[2];
			//OpenCV store pixels as BGR.
		
		}
	}


	// upload cpu data to gpu memory
	depthRaw_.upload(depthData, width*2, height, width);
	color_.upload(imageData, width*3, height, width);

	// convert from column major (default) to row major 
	Eigen::Matrix<float, 3, 3, Eigen::RowMajor> rotMatrix;
	for (int ix = 0; ix < 3; ix++)
	{
		for (int iy = 0; iy < 3; iy++)
		{
			rotMatrix(ix, iy) = frameInfo.rotMatUnscaled(ix, iy);
		}
	}
	Eigen::Matrix<float, 3, 3, Eigen::RowMajor> rotMatrixInverse = rotMatrix.inverse();
	Eigen::Vector3f transVec;
	for (int ix = 0; ix < 3; ix++)
		transVec(ix) = frameInfo.transVec(ix);

	float3 device_volume_size = device_cast<const float3> (tsdfV.getSize());
	
	Mat33&  device_Rcam = device_cast<Mat33> (rotMatrix);
	float3& device_tcam = device_cast<float3> (transVec);
	Mat33&   device_Rcam_inv = device_cast<Mat33> (rotMatrixInverse);

	// static tsdf volume
	
	int3 emptyVoxel;
	emptyVoxel.x = 0;
	emptyVoxel.y = 0;
	emptyVoxel.z = 0;
	
	

	// aggregate information to tsdf data
	printf("aggregating information to tsdf volume ... \n");

	integrateTsdfVolume(depthRaw_,
		frameInfo.intr, 
		device_volume_size,
		device_Rcam_inv,
		device_tcam,
		tsdfV.getTsdfTruncDist(),
		tsdfV.data(),
		depthRawScaled_,
		emptyVoxel,
		colorV.data(),
		color_);

	

	printf("extracting point cloud ... \n");
	// extract and update the reconstructed point cloud
	const Eigen::Vector3i volumeSize = tsdfV.getResolution();
	cloud_buffer_ = tsdfV.fetchCloud(cloud_device_,
		emptyVoxel,
		colorV.data(),
		0, volumeSize(0),
		0, volumeSize(1),
		0, volumeSize(2),
		emptyVoxel);
	/*size_t size= cloud_buffer_.size();
	size_t size2= cloud_device_.size();
	std::cout << "buffer size: " << size << endl;
	std::cout << "device size: " << size2 << endl;*/
	//DeviceArray<pcl::PointXYZRGB> newBuffer(cloud_device_.ptr(), width*height);	
	//std::cout << "new buffer size: " << newBuffer.size () << endl;
	
	printf("downloading to cpu data ... \n");
	cloud_buffer_.download(cloudRec->points);
	//newBuffer.download(cloudRec->points);
	//cloud_device_.download(cloudRec->points);
	//std::cout << "rec size: " << cloudRec->points.size () << endl;
	cloudRec->width = (int)cloudRec->points.size ();
	cloudRec->height = 1;
	//std::cout << "rec size after: " << cloudRec->points.size () << endl;
	printf("KF processed ... \n ");
		
	inDepth.close();
	delete idepth;
	
}





void EvoRec::generatePointCloud(std::string depthFile, std::string colorFile, const FrameInfo& frameInfo)
{
	pcl::PointCloud<pcl::PointXYZRGB> tmpPointCloud;
	std::vector<pcl::PointXYZRGB> tmpPointXYZRGB;
	tmpPointXYZRGB.clear();

	Eigen::Vector3f transVec = frameInfo.transVec;
	Eigen::Matrix3f rotMat = frameInfo.rotMatUnscaled;
	Intr intr = frameInfo.intr;

    std::ifstream inDepth(depthFile.c_str(), std::ios::binary);	
	float* idepth = new float[width*height];
	inDepth.read((char*)idepth, sizeof(float)* width * height);

	cv::Mat imageRGB;
	imageRGB =  cv::imread(colorFile.c_str(), CV_LOAD_IMAGE_COLOR);
	imageRGB.convertTo(imageRGB, CV_8UC3);


	int validNum = 0;
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			int idx = x + y * width;
			// filter some invalid points
			float depth =  idepth[idx]/1000.0f;
			//if (depth > depthThres || depth > MAX_DEPTH || depth < 0)
			//	continue;
			
			
			pcl::PointXYZRGB pointIntoVec;
			float xtmp = (x + 0.5 - intr.cx)*(1.0 / intr.fx) * depth;
			float ytmp = (y + 0.5 - intr.cy)*(1.0 / intr.fy) * depth;
			float ztmp = depth;
			Eigen::Vector3f pointCam(xtmp, ytmp, ztmp);
			Eigen::Vector3f pointWorld = rotMat * pointCam + transVec;
			
			pointIntoVec.x = pointWorld(0);
			pointIntoVec.y = pointWorld(1);
			pointIntoVec.z = pointWorld(2);
			pointIntoVec.b = (unsigned char)imageRGB.at<cv::Vec3b>(y,x)[0];
			pointIntoVec.g = (unsigned char)imageRGB.at<cv::Vec3b>(y,x)[1];
			pointIntoVec.r = (unsigned char)imageRGB.at<cv::Vec3b>(y,x)[2];

			tmpPointXYZRGB.push_back(pointIntoVec);
			validNum++;
		}
	}
    pcl::PointCloud<pcl::PointXYZRGB> cloudOutput;
	cloudOutput.width = validNum;
	cloudOutput.height = 1;
	cloudOutput.points.resize(validNum);
	cloudOutput.is_dense = false;
	for (int i = 0; i < validNum; i++)
	{
		cloudOutput.points[i] = tmpPointXYZRGB[i];
	}
        //pcl::io::savePCDFile ("raw_cloud.pcd", cloudOutput);
        pcl::PCDWriter writer;
        writer.write ("raw_cloud.pcd", cloudOutput, false);
		
	inDepth.close();
	delete idepth;

}

void EvoRec::savePointCloud(std::string outputFileName)
{
	pcl::io::savePCDFile(outputFileName, *cloudRec, false);
}
