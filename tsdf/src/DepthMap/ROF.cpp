#include "ROF.h"

ROF::ROF(int _width, int _height, int _maxIter, float _lambda, float _epsilon)
: width(_width), height(_height), maxIter(_maxIter), lambda(_lambda), epsilon(_epsilon)
{
	initialize();
}

ROF::~ROF()
{
	delete qCurrent;
	delete qNext;
	delete dCurrent;
	delete dNext;
	delete aCurrent;
	delete aNext;
	delete pixelMask;
	delete pixelWeight;
}

void ROF::setMask(bool* _pixelMask)
{
	bool* maxPt = pixelMask + width * height;
	for (bool* pt = pixelMask; pt < maxPt; pt++)
	{
		*pt = *_pixelMask;
		_pixelMask++;
	}
}

void ROF::setWeight(float* _pixelWeight)
{
	float* maxPt = pixelWeight + width * height;
	for (float* pt = pixelWeight; pt < maxPt; pt++)
	{
		*pt = *_pixelWeight;
		_pixelWeight++;
	}
}

void ROF::setGradientWeight(float* _pixelGradientWeight)
{
	float* maxPt = pixelGradientWeight + width * height;
	for (float* pt = pixelGradientWeight; pt < maxPt; pt++)
	{
		*pt = *_pixelGradientWeight;
		_pixelGradientWeight++;
	}
}

void ROF::initialize()
{
	qCurrent = new float[2 * width * height];
	qNext = new float[2 * width * height];
	dCurrent = new float[width * height];
	dNext = new float[width * height];
	aCurrent = new float[width * height];
	// aNext is unused 
	aNext = new float[width * height];

	pixelGradientWeight = new float[width*height];
	pixelWeight = new float[width * height];
	pixelMask = new bool[width * height];

	// set default pixel weight and pixelmask
	float* maxPtGW = pixelGradientWeight + width * height;
	for (float* pt = pixelGradientWeight; pt < maxPtGW; pt++)
	{
		*pt = 1.0f;
	}

	float* maxPtW = pixelWeight + width * height;
	for (float* pt = pixelWeight; pt < maxPtW; pt++)
	{
		*pt = 1.0f;
	}

	bool* maxPtM = pixelMask + width * height;
	for (bool* pt = pixelMask; pt < maxPtM; pt++)
	{
		*pt = false;
	}
}

void ROF::convertFloat2UnsignedChar(unsigned char* dst, const float* src)
{
	unsigned char* maxPt = dst + width * height;
	for (unsigned char* pt = dst; pt < maxPt; pt++)
	{
		*pt = *src;
		src++;
	}
}

void ROF::convertUnsignedChar2Float(float* dst, const unsigned char* src)
{
	float* maxPt = dst + width * height;
	for (float* pt = dst; pt < maxPt; pt++)
	{
		*pt = *src;
		src++;
	}
}

void ROF::denoise(float* dst, const float* src)
{
	// set initial value of sigmaQ and sigmaD
	sigmaQ = 4 / 64.0 / 0.02f;
	sigmaD = 0.02f;
	gamma = 0.7f * lambda;
	// set initial value
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			int idx = x + y * width;
			dCurrent[x + y * width] = src[x + y * width];
			aCurrent[x + y * width] = src[x + y * width];
			qCurrent[idx * 2] = 0.0f;
			qCurrent[idx * 2 + 1] = 0.0f;
		}
	}

	// begin iteration
	for (int iter = 0; iter < maxIter; iter++)
	{
		updateADDenoise();
		updateAtQDenoise();
		updateDQ();
		updateSigmaDQ();
	}

	// copy dCurrent to dst and return
	finalise(dst);
}

void ROF::updateSigmaDQ()
{
	float tupdate = 1.0f / sqrtf(1.0f + 2 * gamma * sigmaD);
	sigmaD *= tupdate;
	sigmaQ /= tupdate;
	if (sigmaD < 0.001f) sigmaD = 0.001f;
	if (sigmaQ > 100.0f) sigmaQ = 100.0f;
}

void ROF::denoise(unsigned char* dst, const float* src)
{
	float* dstFloat = new float[width * height];
	denoise(dstFloat, src);
	convertFloat2UnsignedChar(dst, dstFloat);
	delete dstFloat;
}

void ROF::denoise(unsigned char* dst, const unsigned char* src)
{
	float* srcFloat = new float[width * height];
	float* dstFloat = new float[width * height];
	convertUnsignedChar2Float(srcFloat, src);
	denoise(dstFloat, srcFloat);
	convertFloat2UnsignedChar(dst, dstFloat);
	delete dstFloat;
	delete srcFloat;
}

void ROF::denoise(float* dst, const unsigned char* src)
{
	float* srcFloat = new float[width * height];
	convertUnsignedChar2Float(srcFloat, src);
	denoise(dst, srcFloat);
	delete srcFloat;
}

void ROF::updateDQ()
{
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			int idx = x + y * width;
			dCurrent[idx] = dNext[idx];
			qCurrent[idx * 2] = qNext[idx * 2];
			qCurrent[idx * 2 + 1] = qNext[idx * 2 + 1];
		}
	}
}

void ROF::finalise(float* dst)
{
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			int idx = x + y * width;
			dst[idx] = dCurrent[idx];
		}
	}
}

float ROF::computeDual()
{
  float res = 0.0f;
  for (int y = 0; y < height; y++)
  {
    for (int x = 0; x < width; x++)
    {
      int idx = x + y * width;
	  float weight = pixelWeight[idx];
      float adx = 0.0f;
      if (x < width - 1) adx = dCurrent[idx + 1] - dCurrent[idx];
      float ady = 0.0f;
      if (y < height - 1) ady = dCurrent[idx + width] - dCurrent[idx];
	  if (pixelMask[idx])
		  res += adx * qCurrent[2 * idx] + ady * qCurrent[2 * idx + 1];
	  else
		  res += adx * qCurrent[2 * idx] + ady * qCurrent[2 * idx + 1] +
			  weight * lambda * (dCurrent[idx] - aCurrent[idx]) * (dCurrent[idx] - aCurrent[idx]);
    }
  }
  return res; 
}

float ROF::computePrimal()
{
  float res = 0.0f;
  for (int y = 0; y < height; y++)
  {
    for (int x = 0; x < width; x++)
    {
      int idx = x + y * width;
	  float weight = pixelWeight[idx];
      float adx = 0.0f;
      if (x < width - 1) adx = dCurrent[idx + 1] - dCurrent[idx];
      float ady = 0.0f;
      if (y < height - 1) ady = dCurrent[idx + width] - dCurrent[idx];

	  if (pixelMask[idx])
		  res += sqrtf(adx * adx + ady * ady);
	  else
	      res += sqrtf(adx * adx + ady * ady) + weight * lambda * (dCurrent[idx] - aCurrent[idx]) * (dCurrent[idx] - aCurrent[idx]);
    }
  }
  return res; 
}

void ROF::updateADDenoise()
{
  for (int y = 0; y < height; y++)
  {
    for (int x = 0; x < width; x++)
    {
      int idx = x + y * width;
	  float weightGradient = pixelGradientWeight[idx];
      float adx;
      if (x < width - 1)
      {
        adx = dCurrent[idx + 1] - dCurrent[idx];
      }
      else
      {
        adx = 0.0f;
      }
      float ady;
      if (y < height - 1)
      {
        ady = dCurrent[idx + width] - dCurrent[idx];
      }
      else
      {
        ady = 0.0f;
      }

	  qNext[idx * 2] = qCurrent[idx * 2] + sigmaQ * adx * weightGradient - epsilon * qCurrent[idx * 2];
      // scale += qNext[idx*2] * qNext[idx*2];
	  qNext[idx * 2 + 1] = qCurrent[idx * 2 + 1] + sigmaQ * ady * weightGradient - epsilon * qCurrent[idx * 2 + 1];
      float scale = sqrtf(qNext[idx * 2]*qNext[idx * 2] + qNext[idx * 2 + 1] * qNext[idx * 2 + 1]);
      if (scale > 1.0f)
      {
        qNext[idx * 2] = qNext[idx * 2] / scale;
        qNext[idx * 2 + 1] = qNext[idx * 2 + 1] / scale;
      }
    }
  }
}

void ROF::updateAtQDenoise()
{
  for (int y = 0; y < height; y++)
  {
    for (int x = 0; x < width; x++)
    {
      int idx = x + y * width; 
      float atq  = 0.0f; 
      if (y > 0) atq += (qNext[(idx-width)*2 + 1] * pixelGradientWeight[idx-width]); 
      if (y < height - 1) atq -= (qNext[idx*2 + 1] * pixelGradientWeight[idx]);
      if (x > 0) atq += (qNext[(idx-1)*2] * pixelGradientWeight[idx-1]);
      if (x < width - 1) atq -= (qNext[idx*2] * pixelGradientWeight[idx]);

	  if (pixelMask[idx])
		  dNext[idx] = dCurrent[idx] - sigmaD * atq;
	  else
	      dNext[idx] = ( dCurrent[idx] + sigmaD * ( -atq + 2.0f * lambda * pixelWeight[idx] * aCurrent[idx]) )  
			/ (1.0f + 2.0f * lambda * pixelWeight[idx] * sigmaD); 
    }
  }
}