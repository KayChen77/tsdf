
#include "DenseMap.h"

DenseMap::DenseMap(int _width, int _height, bool _doBilateralFilter, const float _idepthMax,
	const float _idepthMin, const int _fillHoleTestNum, const float _idepthVarMax, const float _maxIntensityError)
	: width(_width), height(_height), doBilateralFilter(_doBilateralFilter), idepthMax(_idepthMax), idepthMin(_idepthMin),
		fillHoleTestNum(_fillHoleTestNum), idepthVarMax(_idepthVarMax), maxIntensityError(_maxIntensityError)
{
	return;
}

DenseMap::~DenseMap()
{
	return;
}

int DenseMap::getdir(std::string dir, std::vector<std::string> &files)
{
	DIR *dp;
	struct dirent *dirp;
	if ((dp = opendir(dir.c_str())) == NULL)
	{
		return -1;
	}
	while ((dirp = readdir(dp)) != NULL)
	{
		std::string name = std::string(dirp->d_name);

		if (name != "." && name != "..")
			files.push_back(name);
	}
	closedir(dp);
	std::sort(files.begin(), files.end());
	if (dir.at(dir.length() - 1) != '/') dir = dir + "/";
	for (unsigned int i = 0; i < files.size(); i++)
	{
		if (files[i].at(0) != '/')
			files[i] = dir + files[i];
	}
	return files.size();
}

void DenseMap::setDepthImage(unsigned char* imageIdepth, const float* idepth)
{
	// printf("seting depth image ... \n");
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			int idx = x + y * width;
			if (idepth[idx] <= idepthMin)
			{
				imageIdepth[idx] = 0;
				continue;
			}
			if (idepth[idx] >= idepthMax)
			{
				imageIdepth[idx] = 255;
				continue;
			}
			imageIdepth[idx] = (idepth[idx] - idepthMin) / (idepthMax - idepthMin) * 255.0f;
		}
	}
}

float DenseMap::getInterpolatedElement(const unsigned char* image, const float x, const float y)
{
	int ix = (int)x;
	int iy = (int)y;
	float dx = x - ix;
	float dy = y - iy;
	float dxdy = dx*dy;
	const unsigned char* bp = image + ix + iy*width;
	float res = dxdy * bp[1 + width]
		+ (dy - dxdy) * bp[width]
		+ (dx - dxdy) * bp[1]
		+ (1 - dx - dy + dxdy) * bp[0];
	return res;
}

void DenseMap::loadKFData(std::string fileName, std::vector<FrameInfo>& KFInfo)
{
	KFInfo.clear();

	std::ifstream inFile(fileName.c_str(), std::ios::binary);
	float* tmpBuffer = new float[width * height * 3];

	FrameInfo Kftmp;
	printf("file opened for loading key frame data ... \n");
	while (!inFile.eof())
	{
		Kftmp.id = -1;
		inFile.read((char*)&Kftmp, sizeof(FrameInfo));
		printf("loading data from kf id %d ... \n", Kftmp.id);
		if (Kftmp.id == -1) break;
		inFile.read((char*)tmpBuffer, sizeof(float)* width * height);
		inFile.read((char*)tmpBuffer, sizeof(float)* width * height);
		inFile.read((char*)tmpBuffer, sizeof(float)* width * height * 3);
		KFInfo.push_back(Kftmp);
	}

	inFile.close();
	delete tmpBuffer;
	printf("Key Frame track data loaded ... \n");

}

int DenseMap::toCamera(const Eigen::Vector3f& Wxp)
{
	float wxp_depth = sqrtf(Wxp(0)*Wxp(0) + Wxp(1)*Wxp(1) + Wxp(2)*Wxp(2));
	float phi = acosf(Wxp(2) / (1e-10f + wxp_depth));
	float theta = acosf(Wxp(0) / (1e-10f + sqrtf(Wxp(0)*Wxp(0) + Wxp(1)*Wxp(1))));
	if (Wxp[1] < 0)
		theta = 2 * M_PI - theta;
	int u = (int)(0.5f + (2.0f * M_PI - theta) / M_PI / 2.0f*width);
	int v = (int)(phi / M_PI*height + 0.5f);

	if (u < 0 || u > width - 1 || v < 0 || v > height - 1) return -1;
	return u + v * width;
}

Eigen::Vector2f DenseMap::KF2F(int x, int y, float idepth, Eigen::Matrix3f& KF2F_rotMat, Eigen::Vector3f& KF2F_transVec)
{
	float theta = 2.0f*x*M_PI / width;
	theta = 2.0f*M_PI - theta;
	float phi = 1.0f*y*M_PI / height;
	float x_sphere = cosf(theta) * sinf(phi);
	float y_sphere = sinf(theta) * sinf(phi);
	float z_sphere = cosf(phi);

	float x_sphere_key = KF2F_rotMat(0, 0) * x_sphere + KF2F_rotMat(0, 1) * y_sphere + KF2F_rotMat(0, 2) * z_sphere + idepth * KF2F_transVec(0);
	float y_sphere_key = KF2F_rotMat(1, 0) * x_sphere + KF2F_rotMat(1, 1) * y_sphere + KF2F_rotMat(1, 2) * z_sphere + idepth * KF2F_transVec(1);
	float z_sphere_key = KF2F_rotMat(2, 0) * x_sphere + KF2F_rotMat(2, 1) * y_sphere + KF2F_rotMat(2, 2) * z_sphere + idepth * KF2F_transVec(2);

	float theta_key = atanf(y_sphere_key / (x_sphere_key + DIVISION_EPS));
	float r_sphere_key = sqrtf(x_sphere_key * x_sphere_key + y_sphere_key * y_sphere_key + z_sphere_key * z_sphere_key);
	float phi_key = acosf(z_sphere_key / (r_sphere_key + DIVISION_EPS));
	if (x_sphere_key < 0)
	{
		theta_key = theta_key + M_PI;
	}
	if (theta_key < 0)
	{
		theta_key = 2 * M_PI + theta_key;
	}

	float u_key = (2.0f * M_PI - theta_key) / M_PI / 2.0f * width;
	float v_key = phi_key / M_PI * height;
	if (u_key > width - 1) u_key = width - 1;
	if (v_key > height - 1) v_key = height - 1;

	return Eigen::Vector2f(u_key, v_key);
}


void DenseMap::fillHolesCostMin(const float* idepthOrg, float* idepthFilled, float* idepthVar, bool* idepthMask, 
	const unsigned char* imageKF, const unsigned char* imageStereo, Eigen::Matrix3f& KF2F_rotMat, Eigen::Vector3f& KF2F_transVec)
{
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			int idx = x + y * width;
			if (idepthOrg[idx] > 0.0f && idepthVar[idx] < idepthVarMax)
			{
				// printf("imageValue: %f ... \n", imageOrg[idx]);
				idepthFilled[idx] = idepthOrg[idx];
				idepthMask[idx] = false;
				continue;
			}
			float bestIdepth = 0.0f;
			float bestCost = INFINITY;
			if (idepthOrg[idx] > 0.0f)
			{
				Eigen::Vector2f uv = KF2F(x, y, idepthOrg[idx], KF2F_rotMat, KF2F_transVec);
				float projectedColor = getInterpolatedElement(imageStereo, uv(0), uv(1));
				float orgColor = imageKF[idx];
				float costTmp = fabsf(projectedColor - orgColor);
				bestCost = costTmp;
				bestIdepth = idepthOrg[idx];
			}

			idepthMask[idx] = true;
			for (int idTest = 0; idTest < fillHoleTestNum; idTest++)
			{
				float idepthRand = idepthMin + (idepthMax - idepthMin) * (rand() % 1000) / 1000.0f;
				Eigen::Vector2f uv = KF2F(x, y, idepthRand, KF2F_rotMat, KF2F_transVec);
				float projectedColor = getInterpolatedElement(imageStereo, uv(0), uv(1));
				float orgColor = imageKF[idx];
				float costTmp = fabsf(projectedColor - orgColor);
				if (costTmp < bestCost)
				{
					bestCost = costTmp;
					bestIdepth = idepthRand;
				}
			}
			idepthFilled[idx] = bestIdepth;
			if (bestCost < maxIntensityError)
				idepthVar[idx] = idepthVarMax - 0.0001f;
			else
				idepthVar[idx] = idepthVarMax + 0.0001f;

		}
	}
}

void DenseMap::getDepthWeight(const float* idepthVarKF, float* idepthWeight)
{
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			int idx = x + y * width;
			if (idepthVarKF[idx] < 0.0f)
				idepthWeight[idx] = 0.0f;
			else
			{
				idepthWeight[idx] = 1.0f / (1.0f + idepthVarKF[idx]);
			}
		}
	}
}

void DenseMap::getPixelGradientWeight(const unsigned char* image, float* pixelGradientWeight)
{
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			float dx, dy;
			int idx = x + y * width;
			if (x == 0)
				dx = image[idx + 1] - image[idx];
			else if (x == width - 1)
				dx = image[idx] - image[idx - 1];
			else
				dx = (image[idx + 1] - image[idx - 1]) / 2.0f;
			if (y == 0)
				dy = image[idx + width] - image[idx];
			else if (y == height - 1)
				dy = image[idx] - image[idx - width];
			else
				dy = (image[idx + width] - image[idx - width]) / 2.0f;

			float gradMagnitude = dx*dx + dy*dy;
			float alpha = 0.2f;
			float beta = 0.7f;
			pixelGradientWeight[idx] = expf(-alpha * std::pow(gradMagnitude, beta));
		}
	}
}

void DenseMap::fillHolesVar(float* iDepthVarKF, float defaultVar)
{
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			int idx = x + y*width;
			if (iDepthVarKF[idx] < 0.0f)
				iDepthVarKF[idx] = defaultVar;
		}
	}
}
void DenseMap::processEvoSlamData(std::string inKFFileName, std::string ImageFileName, std::string outKFFileName)
{
	// prepare rof instance
	ROF rofFilter(width, height);

	// get image name file list
	std::vector<std::string> files;
	getdir(ImageFileName.c_str(), files);

	// load KF track data
	std::vector<FrameInfo> KFInfo;
	loadKFData(inKFFileName, KFInfo);

	// open in and output file
	std::ifstream inFileKF(inKFFileName.c_str(), std::ios::binary);
	std::ofstream outFileKF(outKFFileName.c_str(), std::ios::binary);

	float* iDepthKF = new float[width*height];
	float* iDepthVarKF = new float[width*height];
	float* imageRGBKF = new float[width*height * 3];

	float* iDepthKFFilled = new float[width * height];
	float* iDepthDenoised = new float[width * height];
	bool* idepthMask = new bool[width*height];
	float* idepthWeight = new float[width * height];
	float* pixelGradientWeight = new float[width*height];

	FrameInfo Kftmp;

	int count = 0;
	while (!inFileKF.eof())
	{
		Kftmp.id = -1;
		inFileKF.read((char*)&Kftmp, sizeof(FrameInfo));
		printf("loading data from kf id: %d ... \n", Kftmp.id);
		if (Kftmp.id == -1) break;
		inFileKF.read((char*)iDepthKF, sizeof(float)* width * height);
		inFileKF.read((char*)iDepthVarKF, sizeof(float)* width * height);
		inFileKF.read((char*)imageRGBKF, sizeof(float)* width * height * 3);

		outFileKF.write((char*)&Kftmp, sizeof(FrameInfo));

		// load scaled rotmat and transvec
		Eigen::Matrix3f KF2W_rotMat = KFInfo[count].rotMatUnscaled * KFInfo[count].scale;
		Eigen::Vector3f KF2W_transVec = KFInfo[count].transVec;

		// load kf image
		cv::Mat imageKF = cv::Mat(height, width, CV_8U);
		imageKF = cv::imread(files[KFInfo[count].id], CV_LOAD_IMAGE_GRAYSCALE);
		
		// printf("image loaded ... \n");

		cv::Mat imageKF_filtered = cv::Mat(height, width, CV_8U);

		// load image for volume computation
		cv::Mat image = cv::Mat(height, width, CV_8U);

		cv::Mat image_filtered = cv::Mat(height, width, CV_8U);

		int d = 3;
		double sigmaColor = 2.0;
		double sigmaSpace = 2.0;
		if (doBilateralFilter)
			cv::bilateralFilter(imageKF, imageKF_filtered, d, sigmaColor, sigmaSpace);

		// printf("bilateral filtered for kf... \n");
		if (count == 0)
		{
			image = cv::imread(files[KFInfo[count + 1].id], CV_LOAD_IMAGE_GRAYSCALE);
			if (doBilateralFilter)
				cv::bilateralFilter(image, image_filtered, d, sigmaColor, sigmaSpace);

			// printf("bilateral filtered for stereo kf... \n");

			FrameInfo Ftmp = KFInfo[count + 1];
			Eigen::Matrix3f W2F_rotMat = 1.0f / Ftmp.scale * Ftmp.rotMatUnscaled.transpose();
			Eigen::Vector3f W2F_transVec = -W2F_rotMat * Ftmp.transVec;
			Eigen::Matrix3f KF2F_rotMat = W2F_rotMat * KF2W_rotMat;
			Eigen::Vector3f KF2F_transVec = W2F_rotMat * KF2W_transVec + W2F_transVec;
			
			if (doBilateralFilter)
				fillHolesCostMin(iDepthKF, iDepthKFFilled, iDepthVarKF, idepthMask, imageKF_filtered.data, image_filtered.data, KF2F_rotMat, KF2F_transVec);
			else
				fillHolesCostMin(iDepthKF, iDepthKFFilled, iDepthVarKF, idepthMask, imageKF.data, image.data, KF2F_rotMat, KF2F_transVec);
		}
		else
		{
			image = cv::imread(files[KFInfo[count - 1].id], CV_LOAD_IMAGE_GRAYSCALE);
			if (doBilateralFilter)
				cv::bilateralFilter(image, image_filtered, d, sigmaColor, sigmaSpace);
			FrameInfo Ftmp = KFInfo[count - 1];
			Eigen::Matrix3f W2F_rotMat = 1.0f / Ftmp.scale * Ftmp.rotMatUnscaled.transpose();
			Eigen::Vector3f W2F_transVec = -W2F_rotMat * Ftmp.transVec;
			Eigen::Matrix3f KF2F_rotMat = W2F_rotMat * KF2W_rotMat;
			Eigen::Vector3f KF2F_transVec = W2F_rotMat * KF2W_transVec + W2F_transVec;
			if (doBilateralFilter)
				fillHolesCostMin(iDepthKF, iDepthKFFilled, iDepthVarKF, idepthMask, imageKF_filtered.data, image_filtered.data, KF2F_rotMat, KF2F_transVec);
			else
				fillHolesCostMin(iDepthKF, iDepthKFFilled, iDepthVarKF, idepthMask, imageKF.data, image.data, KF2F_rotMat, KF2F_transVec);
		}

		printf("holes filled for kf id %d... \n", Kftmp.id);
		count++;
		// write image for debug
		cv::Mat imageToWrite = cv::Mat(height, width, CV_8U);
		std::stringstream ss;
		ss << count;
		setDepthImage(imageToWrite.data, iDepthKF);
		cv::imwrite("/home/evocloud/workspace_huang/EvoRecData/DepthOrg" + ss.str() + ".jpg", imageToWrite);
		setDepthImage(imageToWrite.data, iDepthKFFilled);
		cv::imwrite("/home/evocloud/workspace_huang/EvoRecData/DepthFilled" + ss.str() + ".jpg", imageToWrite);

		getDepthWeight(iDepthVarKF, idepthWeight);
		getPixelGradientWeight(imageKF.data, pixelGradientWeight);

		rofFilter.setMask(idepthMask);
		rofFilter.setWeight(idepthWeight);
		rofFilter.setGradientWeight(pixelGradientWeight);

		rofFilter.denoise(iDepthDenoised, iDepthKFFilled);

		setDepthImage(imageToWrite.data, iDepthDenoised);
		cv::imwrite("/home/evocloud/workspace_huang/EvoRecData/DepthDenoised" + ss.str() + ".jpg", imageToWrite);

		// fill idepthVarKF
		fillHolesVar(iDepthVarKF, 1.0f);

		outFileKF.write((char*)iDepthDenoised, sizeof(float)* width * height);
		outFileKF.write((char*)iDepthVarKF, sizeof(float)* width * height);
		outFileKF.write((char*)imageRGBKF, sizeof(float)* width * height * 3);
	}

	delete iDepthKF;
	delete iDepthVarKF;
	delete imageRGBKF;
	delete iDepthKFFilled;
	delete idepthMask;
	delete iDepthDenoised;
	delete idepthWeight;
	delete pixelGradientWeight;

	outFileKF.close();
}
