#pragma once
#include "./internal.h"

struct FrameInfo
{
	int id;
	// F2W
	float scale;
	Eigen::Matrix3f rotMatUnscaled;
	Eigen::Vector3f transVec;
	Intr intr;
};
