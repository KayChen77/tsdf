// created by czr, 19.05.2016

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/console/parse.h>

#include <pcl/kdtree/kdtree_flann.h>

// surface smoothing
#include <pcl/surface/mls.h>

// normal estimation
#include <pcl/features/normal_3d.h>

// filters
#include <pcl/filters/statistical_outlier_removal.h>

// transform
#include <pcl/common/transforms.h>

// registration
#include <pcl/registration/icp.h>

typedef boost::shared_ptr<pcl::visualization::PCLVisualizer> PCLVisualizerPtr;


int main(int argc, char** argv)
{
	// parse console arguments
	std::string file_path = argv[1];

	bool save = false;
	save = pcl::console::find_switch(argc, argv, "--s");

	bool optimal = false;
	optimal = pcl::console::find_switch(argc, argv, "--o");

	// Input obejcts
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_1(new pcl::PointCloud<pcl::PointXYZRGB>());
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_2(new pcl::PointCloud<pcl::PointXYZRGB>());
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_3(new pcl::PointCloud<pcl::PointXYZRGB>());
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_4(new pcl::PointCloud<pcl::PointXYZRGB>());

	// loading from files
	pcl::io::loadPCDFile(file_path + "point_cloud_022718143547_1_filtered.pcd", *cloud_1);
	pcl::io::loadPCDFile(file_path + "point_cloud_023446243547_1_filtered.pcd", *cloud_2);
	pcl::io::loadPCDFile(file_path + "point_cloud_001489661447_1_filtered.pcd", *cloud_3);
	pcl::io::loadPCDFile(file_path + "point_cloud_001496161447_1_filtered.pcd", *cloud_4);

	// objects for direct transform
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_trans_21(new pcl::PointCloud<pcl::PointXYZRGB>());
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_trans_41(new pcl::PointCloud<pcl::PointXYZRGB>());
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_trans_34(new pcl::PointCloud<pcl::PointXYZRGB>());
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_trans_31(new pcl::PointCloud<pcl::PointXYZRGB>());
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_trans_341(new pcl::PointCloud<pcl::PointXYZRGB>());

	// #################################################################
	// ############ Without Optimization for the system ################
	// #################################################################

	if ( optimal == false)
	{
		std::cout << "Use the calibrated transforms" << std::endl;
		// the transform from the kinect 2 to the kinect 1
		Eigen::Matrix4f transform21 = Eigen::Matrix4f::Identity();
		transform21(0, 0) = 0.131596824985008;
		transform21(0, 1) = 0.082978517132829;
		transform21(0, 2) = -0.987824296799943;
		transform21(1, 0) = -0.145470499662941;
		transform21(1, 1) = 0.987319038901770;
		transform21(1, 2) = 0.063556660940451;
		transform21(2, 0) = 0.980571572799089;
		transform21(2, 1) = 0.135335439248270;
		transform21(2, 2) = 0.141998977115325;

		transform21(0, 3) = 1.128942421447028e+03 / 1000.0f;
		transform21(1, 3) = -1.432968237885304e+02 / 1000.0f;
		transform21(2, 3) = 7.893663478974775e+02 / 1000.0f;

		std::cout << "Transform matrix from kinect 2 to kinect 1 obtained in calibration." << std::endl;
		std::cout << transform21 << std::endl;

		// direct transform from 2 to 1
		pcl::transformPointCloud(*cloud_2, *cloud_trans_21, transform21);

		// the transform from the kinect 4 to the kinect 1
		Eigen::Matrix4f transform41 = Eigen::Matrix4f::Identity();
		transform41(0, 0) = -0.077785628871890;
		transform41(0, 1) = -0.156205862982979;
		transform41(0, 2) = 0.984656856123364;
		transform41(1, 0) = 0.159084844028386;
		transform41(1, 1) = 0.973049726937660;
		transform41(1, 2) = 0.166931846293660;
		transform41(2, 0) = -0.984195818087777;
		transform41(2, 1) = 0.169628881020564;
		transform41(2, 2) = -0.050839299584500;

		transform41(0, 3) = -8.272320221254719e+02 / 1000.0f;
		transform41(1, 3) = -1.737206072762551e+02 / 1000.0f;
		transform41(2, 3) = 1.256884515749348e+03 / 1000.0f;

		std::cout << "Transform matrix from kinect 4 to kinect 1 obtained in calibration." << std::endl;
		std::cout << transform41 << std::endl;

		// direct transform from 4 to 1
		pcl::transformPointCloud(*cloud_4, *cloud_trans_41, transform41);

		// the transform from 3 to 1 is obtained by
		// first transform 3 to 4
		Eigen::Matrix4f transform34 = Eigen::Matrix4f::Identity();
		transform34(0, 0) = -0.037043477662306;
		transform34(0, 1) = -0.078694279679547;
		transform34(0, 2) = 0.996210314696851;
		transform34(1, 0) = 0.174838931754836;
		transform34(1, 1) = 0.981007770527207;
		transform34(1, 2) = 0.083994655235126;
		transform34(2, 0) = -0.983899958687625;
		transform34(2, 1) = 0.177287801359702;
		transform34(2, 2) = -0.022581115639685;

		transform34(0, 3) = -8.949660801479120e+02 / 1000.0f;
		transform34(1, 3) = -1.841074499573664e+02 / 1000.0f;
		transform34(2, 3) = 1.001926453924089e+03 / 1000.0f;

		std::cout << "Transform matrix from kinect 3 to kinect 4 obtained in calibration." << std::endl;
		std::cout << transform34 << std::endl;

		pcl::transformPointCloud(*cloud_3, *cloud_trans_34, transform34);

		// second, transform from 4 to 1
		pcl::transformPointCloud(*cloud_trans_34, *cloud_trans_341, transform41);

		// save the resigtered point cloud
		if (save)
		{
			pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_merged(new pcl::PointCloud<pcl::PointXYZRGB>());
			*cloud_merged = *cloud_1;
			*cloud_merged += *cloud_trans_21;
			*cloud_merged += *cloud_trans_41;
			*cloud_merged += *cloud_trans_341;
			pcl::io::savePCDFileASCII("merged_cloud.pcd", *cloud_merged);
			std::cout << "Saved merged point cloud." << std::endl;
		}
	}
	// ##############################################################
	// ############ With Optimization for the system ################
	// ##############################################################

	else //if (optimal == true)
	{
		std::cout << "Use the optimized transforms" << std::endl;

		// prepare for the icps
		// statistical outlier removal
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_sor_1(new pcl::PointCloud<pcl::PointXYZRGB>());
		pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor1;
		sor1.setInputCloud(cloud_1);
		sor1.setMeanK(50);
		sor1.setStddevMulThresh(0.5);
		sor1.setNegative(false);
		sor1.filter(*cloud_sor_1);

		// mls
		pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree_mls_1(new pcl::search::KdTree<pcl::PointXYZRGB>() );
		tree_mls_1->setInputCloud(cloud_sor_1);
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_mls_1(new pcl::PointCloud<pcl::PointXYZRGB>() );
		pcl::MovingLeastSquares<pcl::PointXYZRGB, pcl::PointXYZRGB> mls_1;
		mls_1.setComputeNormals(false);
		mls_1.setInputCloud(cloud_sor_1);
		mls_1.setPolynomialOrder(4);
		mls_1.setPolynomialFit(true);
		mls_1.setSearchMethod(tree_mls_1);
		mls_1.setSearchRadius(0.03);
		mls_1.process(*cloud_mls_1);

		// normal estimation
		pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_with_normals_1(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
		pcl::PointCloud<pcl::Normal>::Ptr normals_1(new pcl::PointCloud<pcl::Normal>());
		pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree_normal_1(new pcl::search::KdTree<pcl::PointXYZRGB>());
		tree_normal_1->setInputCloud(cloud_mls_1);
		pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne_1;
		ne_1.setInputCloud(cloud_mls_1);
		ne_1.setSearchMethod(tree_normal_1);
		ne_1.setKSearch(20);
		ne_1.compute(*normals_1);
		pcl::concatenateFields(*cloud_mls_1, *normals_1, *cloud_with_normals_1);

		pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_ref(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
        *cloud_ref = *cloud_with_normals_1;

		// the transform from the kinect 2 to the kinect 1 (R1,T1)
		Eigen::Matrix4f transform21 = Eigen::Matrix4f::Identity();
		transform21(0, 0) = 0.134495157429655;
		transform21(0, 1) = 0.080498653293076;
		transform21(0, 2) = -0.987639113971279;
		transform21(1, 0) = -0.144135633421401;
		transform21(1, 1) = 0.987683785283996;
		transform21(1, 2) = 0.060874128045401;
		transform21(2, 0) = 0.980375423909730;
		transform21(2, 1) = 0.134166713849142;
		transform21(2, 2) = 0.144441410574456;

		transform21(0, 3) = 1.128942737114943e+03 / 1000.0f;
		transform21(1, 3) = -1.432967440937596e+02 / 1000.0f;
		transform21(2, 3) = 7.893658796324055e+02 / 1000.0f;

		std::cout << "Transform matrix from kinect 2 to kinect 1 obtained in calibration." << std::endl;
		std::cout << transform21 << std::endl;

		// direct transform from 2 to 1
		pcl::transformPointCloud(*cloud_2, *cloud_trans_21, transform21);

		// icp
		pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_icp_21(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
		
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_sor_2(new pcl::PointCloud<pcl::PointXYZRGB>());
		pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor2;
		sor2.setInputCloud(cloud_2);
		sor2.setMeanK(50);
		sor2.setStddevMulThresh(0.5);
		sor2.setNegative(false);
		sor2.filter(*cloud_sor_2);

		pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree_mls_2(new pcl::search::KdTree<pcl::PointXYZRGB>());
		tree_mls_2->setInputCloud(cloud_sor_2);
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_mls_2(new pcl::PointCloud<pcl::PointXYZRGB>());
		pcl::MovingLeastSquares<pcl::PointXYZRGB, pcl::PointXYZRGB> mls_2;
		mls_2.setComputeNormals(false);
		mls_2.setInputCloud(cloud_sor_2);
		mls_2.setPolynomialOrder(4);
		mls_2.setPolynomialFit(true);
		mls_2.setSearchMethod(tree_mls_2);
		mls_2.setSearchRadius(0.03);
		mls_2.process(*cloud_mls_2);
		
		pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_with_normals_2(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
		pcl::PointCloud<pcl::Normal>::Ptr normals_2(new pcl::PointCloud<pcl::Normal>());
		pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree_normal_2(new pcl::search::KdTree<pcl::PointXYZRGB>());
		tree_normal_2->setInputCloud(cloud_mls_2);
		pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne_2;
		ne_2.setInputCloud(cloud_mls_2);
		ne_2.setSearchMethod(tree_normal_2);
		ne_2.setKSearch(20);
		ne_2.compute(*normals_2);
		pcl::concatenateFields(*cloud_mls_2, *normals_2, *cloud_with_normals_2);

		pcl::IterativeClosestPoint<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal> icp_21;
		icp_21.setInputCloud(cloud_with_normals_2);
        icp_21.setInputTarget(cloud_ref);
		icp_21.setMaxCorrespondenceDistance(0.02);
		icp_21.setMaximumIterations(100);
		icp_21.setTransformationEpsilon(1e-8);
		icp_21.setEuclideanFitnessEpsilon(1e-5);
		icp_21.align(*cloud_icp_21, transform21);

		std::cout << "has converged:" << icp_21.hasConverged() << " score: " <<
			icp_21.getFitnessScore() << std::endl;
		std::cout << icp_21.getFinalTransformation() << std::endl;
        
		// for next icp
        *cloud_ref += *cloud_icp_21;

		// the transform from the kinect 4 to the kinect 1 (R3,T3)
		Eigen::Matrix4f transform41 = Eigen::Matrix4f::Identity();
		transform41(0, 0) = -0.073896009032637;
		transform41(0, 1) = -0.155250763238798;
		transform41(0, 2) = 0.985107395344700;
		transform41(1, 0) = 0.158951733970376;
		transform41(1, 1) = 0.973346494691277;
		transform41(1, 2) = 0.165320741408982;
		transform41(2, 0) = -0.984517001436166;
		transform41(2, 1) = 0.168801071637521;
		transform41(2, 2) = -0.047249043346561;

		transform41(0, 3) = -8.272323915033728e+02 / 1000.0f;
		transform41(1, 3) = -1.737204056866098e+02 / 1000.0f;
		transform41(2, 3) = 1.256884341974619e+03 / 1000.0f;

		std::cout << "Transform matrix from kinect 4 to kinect 1 obtained in calibration." << std::endl;
		std::cout << transform41 << std::endl;

		// direct transform from 4 to 1
		pcl::transformPointCloud(*cloud_4, *cloud_trans_41, transform41);

		// icp
		pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_icp_41(new pcl::PointCloud<pcl::PointXYZRGBNormal>());

		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_sor_4(new pcl::PointCloud<pcl::PointXYZRGB>());
		pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor4;
		sor4.setInputCloud(cloud_4);
		sor4.setMeanK(50);
		sor4.setStddevMulThresh(0.5);
		sor4.setNegative(false);
		sor4.filter(*cloud_sor_4);

		pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree_mls_4(new pcl::search::KdTree<pcl::PointXYZRGB>());
		tree_mls_4->setInputCloud(cloud_sor_4);
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_mls_4(new pcl::PointCloud<pcl::PointXYZRGB>());
		pcl::MovingLeastSquares<pcl::PointXYZRGB, pcl::PointXYZRGB> mls_4;
		mls_4.setComputeNormals(false);
		mls_4.setInputCloud(cloud_sor_4);
		mls_4.setPolynomialOrder(4);
		mls_4.setPolynomialFit(true);
		mls_4.setSearchMethod(tree_mls_4);
		mls_4.setSearchRadius(0.03);
		mls_4.process(*cloud_mls_4);

		pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_with_normals_4(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
		pcl::PointCloud<pcl::Normal>::Ptr normals_4(new pcl::PointCloud<pcl::Normal>());
		pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree_normal_4(new pcl::search::KdTree<pcl::PointXYZRGB>());
		tree_normal_4->setInputCloud(cloud_mls_4);
		pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne_4;
		ne_4.setInputCloud(cloud_mls_4);
		ne_4.setSearchMethod(tree_normal_4);
		ne_4.setKSearch(20);
		ne_4.compute(*normals_4);
		pcl::concatenateFields(*cloud_mls_4, *normals_4, *cloud_with_normals_4);

		pcl::IterativeClosestPoint<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal> icp_41;
		icp_41.setInputCloud(cloud_with_normals_4);
        icp_41.setInputTarget(cloud_ref);
		icp_41.setMaxCorrespondenceDistance(0.02);
		icp_41.setMaximumIterations(100);
		icp_41.setTransformationEpsilon(1e-8);
		icp_41.setEuclideanFitnessEpsilon(1e-5);
		icp_41.align(*cloud_icp_41, transform41);

		std::cout << "has converged:" << icp_41.hasConverged() << " score: " <<
			icp_41.getFitnessScore() << std::endl;
		std::cout << icp_41.getFinalTransformation() << std::endl;

		// for next icp
        *cloud_ref += *cloud_icp_41;

		// the transform from 3 to 1 is obtained by (R2,T2)
		Eigen::Matrix4f transform31 = Eigen::Matrix4f::Identity();
		transform31(0, 0) = -0.993943048364607;
		transform31(0, 1) = 0.030325927333675;
		transform31(0, 2) = -0.105629327078354;
		transform31(1, 0) = 0.004398346757531;
		transform31(1, 1) = 0.971379193082247;
		transform31(1, 2) = 0.237493405787788;
		transform31(2, 0) = 0.109808338269340;
		transform31(2, 1) = 0.235590325306951;
		transform31(2, 2) = -0.965628980234278;

		transform31(0, 3) = 2.507968161837474e+02 / 1000.0f;
		transform31(1, 3) = -3.275127035049086e+02 / 1000.0f;
		transform31(2, 3) = 2.057818523590235e+03 / 1000.0f;

		std::cout << "Transform matrix from kinect 3 to kinect 4 obtained in calibration." << std::endl;
		std::cout << transform31 << std::endl;

		// direct transform
		pcl::transformPointCloud(*cloud_3, *cloud_trans_31, transform31);

		// icp
		pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_icp_31(new pcl::PointCloud<pcl::PointXYZRGBNormal>());

		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_sor_3(new pcl::PointCloud<pcl::PointXYZRGB>());
		pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor3;
		sor3.setInputCloud(cloud_3);
		sor3.setMeanK(50);
		sor3.setStddevMulThresh(0.5);
		sor3.setNegative(false);
		sor3.filter(*cloud_sor_3);

		pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree_mls_3(new pcl::search::KdTree<pcl::PointXYZRGB>());
		tree_mls_3->setInputCloud(cloud_sor_3);
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_mls_3(new pcl::PointCloud<pcl::PointXYZRGB>());
		pcl::MovingLeastSquares<pcl::PointXYZRGB, pcl::PointXYZRGB> mls_3;
		mls_3.setComputeNormals(false);
		mls_3.setInputCloud(cloud_sor_3);
		mls_3.setPolynomialOrder(4);
		mls_3.setPolynomialFit(true);
		mls_3.setSearchMethod(tree_mls_3);
		mls_3.setSearchRadius(0.03);
		mls_3.process(*cloud_mls_3);

		pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_with_normals_3(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
		pcl::PointCloud<pcl::Normal>::Ptr normals_3(new pcl::PointCloud<pcl::Normal>());
		pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree_normal_3(new pcl::search::KdTree<pcl::PointXYZRGB>());
		tree_normal_3->setInputCloud(cloud_mls_3);
		pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne_3;
		ne_3.setInputCloud(cloud_mls_3);
		ne_3.setSearchMethod(tree_normal_3);
		ne_3.setKSearch(50);
		ne_3.compute(*normals_3);
		pcl::concatenateFields(*cloud_mls_3, *normals_3, *cloud_with_normals_3);

		pcl::IterativeClosestPoint<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal> icp_31;
		icp_31.setInputCloud(cloud_with_normals_3);
		icp_31.setInputTarget(cloud_ref);
		icp_31.setMaxCorrespondenceDistance(0.02);
		icp_31.setMaximumIterations(100);
		icp_31.setTransformationEpsilon(1e-8);
		icp_31.setEuclideanFitnessEpsilon(1e-5);
		icp_31.align(*cloud_icp_31, transform31);

		std::cout << "has converged:" << icp_31.hasConverged() << " score: " <<
			icp_31.getFitnessScore() << std::endl;
		std::cout << icp_31.getFinalTransformation() << std::endl;

		// result
		*cloud_ref += *cloud_icp_31;

		// save the resigtered point cloud
		if (save)
		{
			pcl::io::savePCDFileASCII("merged_cloud.pcd", *cloud_ref);
			std::cout << "Saved merged point cloud." << std::endl;
		}

	   // visualization 
		pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal> rgb_ref(cloud_ref);

		pcl::visualization::PCLVisualizer viewer("ICP");
		viewer.addCoordinateSystem(1.0);
		viewer.initCameraParameters();
		viewer.setBackgroundColor(0.5, 0.5, 0.5, 0);
		viewer.addPointCloud<pcl::PointXYZRGBNormal>(cloud_ref, rgb_ref, "cloud");

		viewer.spin();
		while (!viewer.wasStopped()) {
			viewer.spinOnce(100);
			usleep(100000);
		}

	}

	// visualize the direct transform
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb1(cloud_1);
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb_trans21(cloud_trans_21);
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb_trans41(cloud_trans_41);
	

	pcl::visualization::PCLVisualizer viewer2("Direct Transform");
	viewer2.addCoordinateSystem(1.0);
	viewer2.initCameraParameters();
	viewer2.setBackgroundColor(0.5, 0.5, 0.5, 0);
	viewer2.addPointCloud<pcl::PointXYZRGB>(cloud_1, rgb1, "cloud_1");
	viewer2.addPointCloud<pcl::PointXYZRGB>(cloud_trans_21, rgb_trans21, "cloud_21");
	viewer2.addPointCloud<pcl::PointXYZRGB>(cloud_trans_41, rgb_trans41, "cloud_41");

	if (optimal == false){
		pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb_trans341(cloud_trans_341);
		viewer2.addPointCloud<pcl::PointXYZRGB>(cloud_trans_341, rgb_trans341, "cloud_31");
	}
	else {
		pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb_trans31(cloud_trans_31);
		viewer2.addPointCloud<pcl::PointXYZRGB>(cloud_trans_31, rgb_trans31, "cloud_31");
	}

	viewer2.spin();
	while (!viewer2.wasStopped()) {
		viewer2.spinOnce(100);
		usleep(100000);
	}

	return (0);

}// end of main()
