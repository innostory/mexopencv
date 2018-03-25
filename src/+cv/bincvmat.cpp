/**
 * @file bincvmat.cpp
 * @brief mex interface for load opencv binary mat file
 * @ingroup core
 */
#include "mexopencv.hpp"

#include <vector>
#include <string.h>

using namespace std;
using namespace cv;

static bool readMatBinary(const Mat& buf, Mat& in_mat)
{
    if (buf.isContinuous() == false || buf.type() != CV_8U) {
        mexErrMsgIdAndTxt("mexopencv:error", "bincvmat input must be uint8 buffer");
        return false;
    }
    const char* b = (const char*)buf.data; 
    const char* p = b;
    int rows = *(int*)p; p += sizeof(int);
    int cols = *(int*)p; p += sizeof(int);
    int type = *(int*)p; p += sizeof(int);
    in_mat.release();
    in_mat.create(rows, cols, type);
    memcpy(in_mat.data, p, b + buf.cols*buf.rows - p);
    return true;
}

/**
 * Main entry called from Matlab
 * @param nlhs number of left-hand-side arguments
 * @param plhs pointers to mxArrays in the left-hand-side
 * @param nrhs number of right-hand-side arguments
 * @param prhs pointers to mxArrays in the right-hand-side
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Check the number of arguments
    nargchk(nrhs==1 && nlhs<=1);

    // Argument vector
    vector<MxArray> rhs(prhs, prhs+nrhs);

    // Process
    Mat img;
    if (readMatBinary(rhs[0].toMat(), img)) {
        plhs[0] = MxArray(img);
    }
}
