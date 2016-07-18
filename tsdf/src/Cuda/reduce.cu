// under review, 3.7.2016

#include "device.hpp"

#if __CUDA_ARCH__ < 300
__inline__ __device__
float __shfl_down(float val, int offset, int width = 32)
{
    static __shared__ float shared[MAX_THREADS];
    int lane = threadIdx.x % 32;
    shared[threadIdx.x] = val;
    __syncthreads();
    val = (lane + offset < width) ? shared[threadIdx.x + offset] : 0;
    __syncthreads();
    return val;
}

__inline__ __device__
int __shfl_down(int val, int offset, int width = 32)
{
    static __shared__ int shared[MAX_THREADS];
    int lane = threadIdx.x % 32;
    shared[threadIdx.x] = val;
    __syncthreads();
    val = (lane + offset < width) ? shared[threadIdx.x + offset] : 0;
    __syncthreads();
    return val;
}
#endif

#if __CUDA_ARCH__ < 350
template<typename T>
__device__ __forceinline__ T __ldg(const T* ptr)
{
    return *ptr;
}
#endif

__inline__  __device__ JtJJtrSE3 warpReduceSum(JtJJtrSE3 val)
{
    for(int offset = warpSize / 2; offset > 0; offset /= 2)
    {
        val.aa += __shfl_down(val.aa, offset);
        val.ab += __shfl_down(val.ab, offset);
        val.ac += __shfl_down(val.ac, offset);
        val.ad += __shfl_down(val.ad, offset);
        val.ae += __shfl_down(val.ae, offset);
        val.af += __shfl_down(val.af, offset);
        val.ag += __shfl_down(val.ag, offset);

        val.bb += __shfl_down(val.bb, offset);
        val.bc += __shfl_down(val.bc, offset);
        val.bd += __shfl_down(val.bd, offset);
        val.be += __shfl_down(val.be, offset);
        val.bf += __shfl_down(val.bf, offset);
        val.bg += __shfl_down(val.bg, offset);

        val.cc += __shfl_down(val.cc, offset);
        val.cd += __shfl_down(val.cd, offset);
        val.ce += __shfl_down(val.ce, offset);
        val.cf += __shfl_down(val.cf, offset);
        val.cg += __shfl_down(val.cg, offset);

        val.dd += __shfl_down(val.dd, offset);
        val.de += __shfl_down(val.de, offset);
        val.df += __shfl_down(val.df, offset);
        val.dg += __shfl_down(val.dg, offset);

        val.ee += __shfl_down(val.ee, offset);
        val.ef += __shfl_down(val.ef, offset);
        val.eg += __shfl_down(val.eg, offset);

        val.ff += __shfl_down(val.ff, offset);
        val.fg += __shfl_down(val.fg, offset);

        val.residual += __shfl_down(val.residual, offset);
        val.inliers += __shfl_down(val.inliers, offset);
    }

    return val;
}

__inline__  __device__ JtJJtrSE3 blockReduceSum(JtJJtrSE3 val)
{
    static __shared__ JtJJtrSE3 shared[32];

    int lane = threadIdx.x % warpSize;

    int wid = threadIdx.x / warpSize;

    val = warpReduceSum(val);

    //write reduced value to shared memory
    if(lane == 0)
    {
        shared[wid] = val;
    }
    __syncthreads();

    const JtJJtrSE3 zero = {0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0};

    //ensure we only grab a value from shared memory if that warp existed
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : zero;

    if(wid == 0)
    {
        val = warpReduceSum(val);
    }

    return val;
}

__global__ void reduceSum(JtJJtrSE3 * in, JtJJtrSE3 * out, int N)
{
    JtJJtrSE3 sum = {0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0};

    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
    {
        sum.add(in[i]);
    }

    sum = blockReduceSum(sum);

    if(threadIdx.x == 0)
    {
        out[blockIdx.x] = sum;
    }
}

struct ICPReduction
{
    Mat33 Rcurr;
    float3 tcurr;

    PtrStep<float> vmap_curr;
    PtrStep<float> nmap_curr;

    Mat33 Rprev_inv;
    float3 tprev;

    Intr intr;

    PtrStep<float> vmap_g_prev;
    PtrStep<float> nmap_g_prev;

    float distThres;
    float angleThres;

    int cols;
    int rows;
    int N;

    JtJJtrSE3 * out;

    __device__ __forceinline__ bool
    search (int & x, int & y, float3& n, float3& d, float3& s) const
    {
        float3 vcurr;
        vcurr.x = vmap_curr.ptr (y       )[x];
        vcurr.y = vmap_curr.ptr (y + rows)[x];
        vcurr.z = vmap_curr.ptr (y + 2 * rows)[x];

		// get coordinate in world system
        float3 vcurr_g = Rcurr * vcurr + tcurr;
		// get coordinate in previous camera system
        float3 vcurr_cp = Rprev_inv * (vcurr_g - tprev);

		// project to previous image (rounded down)
        int2 ukr;
        ukr.x = __float2int_rn (vcurr_cp.x * intr.fx / vcurr_cp.z + intr.cx);
        ukr.y = __float2int_rn (vcurr_cp.y * intr.fy / vcurr_cp.z + intr.cy);

        if(ukr.x < 0 || ukr.y < 0 || ukr.x >= cols || ukr.y >= rows || vcurr_cp.z < 0)
            return false;

		// get previous position
        float3 vprev_g;
        vprev_g.x = __ldg(&vmap_g_prev.ptr (ukr.y       )[ukr.x]);
        vprev_g.y = __ldg(&vmap_g_prev.ptr (ukr.y + rows)[ukr.x]);
        vprev_g.z = __ldg(&vmap_g_prev.ptr (ukr.y + 2 * rows)[ukr.x]);

        float3 ncurr;
        ncurr.x = nmap_curr.ptr (y)[x];
        ncurr.y = nmap_curr.ptr (y + rows)[x];
        ncurr.z = nmap_curr.ptr (y + 2 * rows)[x];

		// transform ncurr to world coordinate
        float3 ncurr_g = Rcurr * ncurr;

        float3 nprev_g;
        nprev_g.x =  __ldg(&nmap_g_prev.ptr (ukr.y)[ukr.x]);
        nprev_g.y = __ldg(&nmap_g_prev.ptr (ukr.y + rows)[ukr.x]);
        nprev_g.z = __ldg(&nmap_g_prev.ptr (ukr.y + 2 * rows)[ukr.x]);

        float dist = norm (vprev_g - vcurr_g);
        float sine = norm (cross (ncurr_g, nprev_g));

		// return the corresponded normal direction, vertex position and current vertex position in world system
        n = nprev_g;
        d = vprev_g;
        s = vcurr_g;

		// return pair found if the angle distance and absolute distance is smaller than the predefiend threshold
        return (sine < angleThres && dist <= distThres && !isnan (ncurr.x) && !isnan (nprev_g.x));
    }

    __device__ __forceinline__ JtJJtrSE3
    getProducts(int & i) const
    {
        int y = i / cols;
        int x = i - (y * cols);

        float3 n_cp, d_cp, s_cp;

        bool found_coresp = search (x, y, n_cp, d_cp, s_cp);

        float row[7] = {0, 0, 0, 0, 0, 0, 0};

        if(found_coresp)
        {
			// now is really vertex position of current point cloud in the previous camera's coordiante system
            s_cp = Rprev_inv * (s_cp - tprev);
			// vertex position of previous point cloud in the previous camera's coordiante system
            d_cp = Rprev_inv * (d_cp - tprev);
			// normal direction of previous point cloud in the previous camera's coordinate system
            n_cp = Rprev_inv * (n_cp);

			// E = (n_cp .* (s_cp - n_cp))^2
			// note that n_cp.* A * eps can be represented as J * eps form which becomes a classic problem of gradient descent
			// d s_cp(eps) d eps = [I W], where W = [0 z -y;-z 0 x;y -x 0];
			// n_cp .* d s_cp(eps) deps = [nx, ny, nz, -z * ny + y * nz, z*nx - x*nz, -y*nx + x*ny]   
			// while the later = cross (s_cp, n_cp), as cross(s_cp, n_cp) = -n_cp * crossMatrix(s_cp)
            *(float3*)&row[0] = n_cp;
            *(float3*)&row[3] = cross (s_cp, n_cp);
            row[6] = dot (n_cp, s_cp - d_cp);
        }

        JtJJtrSE3 values = {row[0] * row[0],
                            row[0] * row[1],
                            row[0] * row[2],
                            row[0] * row[3],
                            row[0] * row[4],
                            row[0] * row[5],
                            row[0] * row[6],

                            row[1] * row[1],
                            row[1] * row[2],
                            row[1] * row[3],
                            row[1] * row[4],
                            row[1] * row[5],
                            row[1] * row[6],

                            row[2] * row[2],
                            row[2] * row[3],
                            row[2] * row[4],
                            row[2] * row[5],
                            row[2] * row[6],

                            row[3] * row[3],
                            row[3] * row[4],
                            row[3] * row[5],
                            row[3] * row[6],

                            row[4] * row[4],
                            row[4] * row[5],
                            row[4] * row[6],

                            row[5] * row[5],
                            row[5] * row[6],

                            row[6] * row[6],
                            found_coresp};

        return values;
    }

    __device__ __forceinline__ void
    operator () () const
    {
        JtJJtrSE3 sum = {0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0};

        for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
        {
            JtJJtrSE3 val = getProducts(i);

            sum.add(val);
        }

        sum = blockReduceSum(sum);

        if(threadIdx.x == 0)
        {
            out[blockIdx.x] = sum;
        }
    }
};

__global__ void icpKernel(const ICPReduction icp)
{
    icp();
}

// icp algorithm 
void icpStep(const Mat33& Rcurr,
             const float3& tcurr,
             const DeviceArray2D<float>& vmap_curr,
             const DeviceArray2D<float>& nmap_curr,
             const Mat33& Rprev_inv,
             const float3& tprev,
             const Intr& intr,
             const DeviceArray2D<float>& vmap_g_prev,
             const DeviceArray2D<float>& nmap_g_prev,
             float distThres,
             float angleThres,
             DeviceArray<JtJJtrSE3> & sum,
             DeviceArray<JtJJtrSE3> & out,
             float * matrixA_host,
             float * vectorB_host,
             float * residual_host,
             int threads,
             int blocks)
{
	// get cols and rows
    int cols = vmap_curr.cols ();
    int rows = vmap_curr.rows () / 3;

	// initialize the icpreduction instance
    ICPReduction icp;

    icp.Rcurr = Rcurr;
    icp.tcurr = tcurr;

    icp.vmap_curr = vmap_curr;
    icp.nmap_curr = nmap_curr;

    icp.Rprev_inv = Rprev_inv;
    icp.tprev = tprev;

    icp.intr = intr;

    icp.vmap_g_prev = vmap_g_prev;
    icp.nmap_g_prev = nmap_g_prev;

    icp.distThres = distThres;
    icp.angleThres = angleThres;

    icp.cols = cols;
    icp.rows = rows;

    icp.N = cols * rows;
    icp.out = sum;

	// call icp kernel function
    icpKernel<<<blocks, threads>>>(icp);

	// call reducesum function
    reduceSum<<<1, MAX_THREADS>>>(sum, out, blocks);

    cudaSafeCall(cudaGetLastError());
    cudaSafeCall(cudaDeviceSynchronize());

	// download to Jt*J, Jt*r
    float host_data[32];
    out.download((JtJJtrSE3 *)&host_data[0]);

	// assign the JtJ and Jtr matrix/vector from host data
    int shift = 0;
    for (int i = 0; i < 6; ++i)
    {
        for (int j = i; j < 7; ++j)
        {
            float value = host_data[shift++];
            if (j == 6)
                vectorB_host[i] = value;
            else
                matrixA_host[j * 6 + i] = matrixA_host[i * 6 + j] = value;
        }
    }

	// assign residual and inliers num
    residual_host[0] = host_data[27];
    residual_host[1] = host_data[28];
}

#define FLT_EPSILON ((float)1.19209290E-07F)

struct RGBReduction
{
    PtrStepSz<DataTerm> corresImg;

    float sigma;
    PtrStepSz<float3> cloud;
    float fx;
    float fy;
    PtrStepSz<short> dIdx;
    PtrStepSz<short> dIdy;
    float sobelScale;

    int cols;
    int rows;
    int N;

    JtJJtrSE3 * out;

    __device__ __forceinline__ JtJJtrSE3
    getProducts(int & i) const
    {
        const DataTerm & corresp = corresImg.data[i];

        bool found_coresp = corresp.valid;

        float row[7];

        if(found_coresp)
        {
            float w = sigma + std::abs(corresp.diff);

            w = w > FLT_EPSILON ? 1.0f / w : 1.0f;

            //Signals RGB only tracking, so we should only
            if(sigma == -1)
            {
                w = 1;
            }

            row[6] = -w * corresp.diff;

            float3 cloudPoint = {cloud.ptr(corresp.zero.y)[corresp.zero.x].x,
                                 cloud.ptr(corresp.zero.y)[corresp.zero.x].y,
                                 cloud.ptr(corresp.zero.y)[corresp.zero.x].z};

            float invz = 1.0 / cloudPoint.z;
            float dI_dx_val = w * sobelScale * dIdx.ptr(corresp.one.y)[corresp.one.x];
            float dI_dy_val = w * sobelScale * dIdy.ptr(corresp.one.y)[corresp.one.x];
            float v0 = dI_dx_val * fx * invz;
            float v1 = dI_dy_val * fy * invz;
            float v2 = -(v0 * cloudPoint.x + v1 * cloudPoint.y) * invz;

            row[0] = v0;
            row[1] = v1;
            row[2] = v2;
            row[3] = -cloudPoint.z * v1 + cloudPoint.y * v2;
            row[4] =  cloudPoint.z * v0 - cloudPoint.x * v2;
            row[5] = -cloudPoint.y * v0 + cloudPoint.x * v1;
        }
        else
        {
            row[0] = row[1] = row[2] = row[3] = row[4] = row[5] = row[6] = 0.f;
        }

        JtJJtrSE3 values = {row[0] * row[0],
                            row[0] * row[1],
                            row[0] * row[2],
                            row[0] * row[3],
                            row[0] * row[4],
                            row[0] * row[5],
                            row[0] * row[6],

                            row[1] * row[1],
                            row[1] * row[2],
                            row[1] * row[3],
                            row[1] * row[4],
                            row[1] * row[5],
                            row[1] * row[6],

                            row[2] * row[2],
                            row[2] * row[3],
                            row[2] * row[4],
                            row[2] * row[5],
                            row[2] * row[6],

                            row[3] * row[3],
                            row[3] * row[4],
                            row[3] * row[5],
                            row[3] * row[6],

                            row[4] * row[4],
                            row[4] * row[5],
                            row[4] * row[6],

                            row[5] * row[5],
                            row[5] * row[6],

                            row[6] * row[6],
                            found_coresp};

        return values;
    }

    __device__ __forceinline__ void
    operator () () const
    {
        JtJJtrSE3 sum = {0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0};

        for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
        {
            JtJJtrSE3 val = getProducts(i);

            sum.add(val);
        }

        sum = blockReduceSum(sum);

        if(threadIdx.x == 0)
        {
            out[blockIdx.x] = sum;
        }
    }
};

__global__ void rgbKernel (const RGBReduction rgb)
{
    rgb();
}

void rgbStep(const DeviceArray2D<DataTerm> & corresImg,
             const float & sigma,
             const DeviceArray2D<float3> & cloud,
             const float & fx,
             const float & fy,
             const DeviceArray2D<short> & dIdx,
             const DeviceArray2D<short> & dIdy,
             const float & sobelScale,
             DeviceArray<JtJJtrSE3> & sum,
             DeviceArray<JtJJtrSE3> & out,
             float * matrixA_host,
             float * vectorB_host,
             int threads,
             int blocks)
{
    RGBReduction rgb;

    rgb.corresImg = corresImg;
    rgb.cols = corresImg.cols();
    rgb.rows = corresImg.rows();
    rgb.sigma = sigma;
    rgb.cloud = cloud;
    rgb.fx = fx;
    rgb.fy = fy;
    rgb.dIdx = dIdx;
    rgb.dIdy = dIdy;
    rgb.sobelScale = sobelScale;
    rgb.N = rgb.cols * rgb.rows;
    rgb.out = sum;

    rgbKernel<<<blocks, threads>>>(rgb);

    reduceSum<<<1, MAX_THREADS>>>(sum, out, blocks);

    cudaSafeCall(cudaGetLastError());
    cudaSafeCall(cudaDeviceSynchronize());

    float host_data[32];
    out.download((JtJJtrSE3 *)&host_data[0]);

    int shift = 0;
    for (int i = 0; i < 6; ++i)
    {
        for (int j = i; j < 7; ++j)
        {
            float value = host_data[shift++];
            if (j == 6)
                vectorB_host[i] = value;
            else
                matrixA_host[j * 6 + i] = matrixA_host[i * 6 + j] = value;
        }
    }
}

__inline__  __device__ int2 warpReduceSum(int2 val)
{
    for(int offset = warpSize / 2; offset > 0; offset /= 2)
    {
        val.x += __shfl_down(val.x, offset);
        val.y += __shfl_down(val.y, offset);
    }

    return val;
}

__inline__  __device__ int2 blockReduceSum(int2 val)
{
    static __shared__ int2 shared[32];

    int lane = threadIdx.x % warpSize;

    int wid = threadIdx.x / warpSize;

    val = warpReduceSum(val);

    //write reduced value to shared memory
    if(lane == 0)
    {
        shared[wid] = val;
    }
    __syncthreads();

    const int2 zero = {0, 0};

    //ensure we only grab a value from shared memory if that warp existed
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : zero;

    if(wid == 0)
    {
        val = warpReduceSum(val);
    }

    return val;
}

__global__ void reduceSum(int2 * in, int2 * out, int N)
{
    int2 sum = {0, 0};

    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
    {
        sum.x += in[i].x;
        sum.y += in[i].y;
    }

    sum = blockReduceSum(sum);

    if(threadIdx.x == 0)
    {
        out[blockIdx.x] = sum;
    }
}

struct RGBResidual
{
    float minScale;

    PtrStepSz<short> dIdx;
    PtrStepSz<short> dIdy;

    PtrStepSz<float> lastDepth;
    PtrStepSz<float> nextDepth;

    PtrStepSz<unsigned char> lastImage;
    PtrStepSz<unsigned char> nextImage;

    mutable PtrStepSz<DataTerm> corresImg;

    float maxDepthDelta;

    float3 kt;
    Mat33 krkinv;

    int cols;
    int rows;
    int N;

    int pitch;
    int imgPitch;

    int2 * out;

    __device__ __forceinline__ int2
    getProducts(int k) const
    {
        int i = k / cols;
        int j0 = k - (i * cols);

        int2 value = {0, 0};

        DataTerm corres;

        corres.valid = false;

        if(i >= 0 && i < rows && j0 >= 0 && j0 < cols)
        {
            if(j0 < cols - 5 && i < rows - 1)
            {
                bool valid = true;

                for(int u = max(i - 2, 0); u < min(i + 2, rows); u++)
                {
                    for(int v = max(j0 - 2, 0); v < min(j0 + 2, cols); v++)
                    {
                        valid = valid && (nextImage.ptr(u)[v] > 0);
                    }
                }

                if(valid)
                {
                    short * ptr_input_x = (short*) ((unsigned char*) dIdx.data + i * pitch);
                    short * ptr_input_y = (short*) ((unsigned char*) dIdy.data + i * pitch);

                    short valx = ptr_input_x[j0];
                    short valy = ptr_input_y[j0];
                    float mTwo = (valx * valx) + (valy * valy);

                    if(mTwo >= minScale)
                    {
                        int y = i;
                        int x = j0;

                        float d1 = nextDepth.ptr(y)[x];

                        if(!isnan(d1))
                        {
                            float transformed_d1 = (float)(d1 * (krkinv.data[2].x * x + krkinv.data[2].y * y + krkinv.data[2].z) + kt.z);
                            int u0 = __float2int_rn((d1 * (krkinv.data[0].x * x + krkinv.data[0].y * y + krkinv.data[0].z) + kt.x) / transformed_d1);
                            int v0 = __float2int_rn((d1 * (krkinv.data[1].x * x + krkinv.data[1].y * y + krkinv.data[1].z) + kt.y) / transformed_d1);

                            if(u0 >= 0 && v0 >= 0 && u0 < lastDepth.cols && v0 < lastDepth.rows)
                            {
                                float d0 = lastDepth.ptr(v0)[u0];

                                if(d0 > 0 && std::abs(transformed_d1 - d0) <= maxDepthDelta && lastImage.ptr(v0)[u0] != 0)
                                {
                                    corres.zero.x = u0;
                                    corres.zero.y = v0;
                                    corres.one.x = x;
                                    corres.one.y = y;
                                    corres.diff = static_cast<float>(nextImage.ptr(y)[x]) - static_cast<float>(lastImage.ptr(v0)[u0]);
                                    corres.valid = true;
                                    value.x = 1;
                                    value.y = corres.diff * corres.diff;
                                }
                            }
                        }
                    }
                }
            }
        }

        corresImg.data[k] = corres;

        return value;
    }

    __device__ __forceinline__ void
    operator () () const
    {
        int2 sum = {0, 0};

        for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
        {
            int2 val = getProducts(i);
            sum.x += val.x;
            sum.y += val.y;
        }

        sum = blockReduceSum(sum);

        if(threadIdx.x == 0)
        {
            out[blockIdx.x] = sum;
        }
    }
};

__global__ void residualKernel (const RGBResidual rgb)
{
    rgb();
}

void computeRgbResidual(const float & minScale,
                        const DeviceArray2D<short> & dIdx,
                        const DeviceArray2D<short> & dIdy,
                        const DeviceArray2D<float> & lastDepth,
                        const DeviceArray2D<float> & nextDepth,
                        const DeviceArray2D<unsigned char> & lastImage,
                        const DeviceArray2D<unsigned char> & nextImage,
                        DeviceArray2D<DataTerm> & corresImg,
                        DeviceArray<int2> & sumResidual,
                        const float maxDepthDelta,
                        const float3 & kt,
                        const Mat33 & krkinv,
                        int & sigmaSum,
                        int & count,
                        int threads,
                        int blocks)
{
    int cols = nextImage.cols ();
    int rows = nextImage.rows ();

    RGBResidual rgb;

    rgb.minScale = minScale;

    rgb.dIdx = dIdx;
    rgb.dIdy = dIdy;

    rgb.lastDepth = lastDepth;
    rgb.nextDepth = nextDepth;

    rgb.lastImage = lastImage;
    rgb.nextImage = nextImage;

    rgb.corresImg = corresImg;

    rgb.maxDepthDelta = maxDepthDelta;

    rgb.kt = kt;
    rgb.krkinv = krkinv;

    rgb.cols = cols;
    rgb.rows = rows;
    rgb.pitch = dIdx.step();
    rgb.imgPitch = nextImage.step();

    rgb.N = cols * rows;
    rgb.out = sumResidual;

	// get the residual sum square to rgb.out.y, num to rgb.out.x
    residualKernel<<<blocks, threads>>>(rgb);

    int2 out_host = {0, 0};
    int2 * out;

    cudaMalloc(&out, sizeof(int2));
    cudaMemcpy(out, &out_host, sizeof(int2), cudaMemcpyHostToDevice);

    reduceSum<<<1, MAX_THREADS>>>(sumResidual, out, blocks);

    cudaSafeCall(cudaGetLastError());
    cudaSafeCall(cudaDeviceSynchronize());

    cudaMemcpy(&out_host, out, sizeof(int2), cudaMemcpyDeviceToHost);
    cudaFree(out);

    count = out_host.x;
    sigmaSum = out_host.y;
}
